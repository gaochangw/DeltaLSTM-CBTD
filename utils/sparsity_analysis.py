"""
Bit-level sparsity analysis and theoretical speedup evaluation for quantized fixed-point weights.

Computes 5 sparsity exploitation scenarios assuming dedicated zero-skipping hardware:
  1. Weight sparsity alone         - Skip MAC where weight element = 0
  2. Temporal sparsity alone       - Skip MAC where delta element = 0
  3. Bit-level weight sparsity     - Skip zero bits in bit-serial weight processing
  4. Weight + Temporal              - Skip MAC where weight = 0 OR delta = 0
  5. Weight + Temporal + Bit-level  - For remaining non-zero MACs, skip zero weight bits

Supports both two's complement and sign-magnitude (BitWave) bit representations.
"""

import numpy as np
import torch

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _to_unsigned_twos_complement(int_weights, bw):
    """Convert signed integers to two's complement unsigned representation."""
    return np.where(int_weights >= 0, int_weights, int_weights + (1 << bw)).astype(np.int64)


def _to_unsigned_sign_magnitude(int_weights, bw):
    """Convert signed integers to sign-magnitude unsigned representation.

    BitWave (HPCA 2025): sign bit separate from magnitude bits.
    For negative values, sign_bit=1 and magnitude = |value|.
    This reduces popcount for small-magnitude negative weights.
    """
    magnitude = np.abs(int_weights).astype(np.int64)
    sign_bit = np.where(int_weights < 0, 1 << (bw - 1), 0).astype(np.int64)
    return sign_bit | magnitude


def _count_nonzero_bits(unsigned_values, bw):
    """Count total non-zero bits across an array of unsigned values."""
    if bw == 8:
        bits_array = np.unpackbits(unsigned_values.astype(np.uint8))
        return int(np.sum(bits_array))
    else:
        nonzero_bits = 0
        for b in range(bw):
            nonzero_bits += int(np.sum((unsigned_values >> b) & 1))
        return nonzero_bits


def compute_bit_level_sparsity(weight_tensor, qi, qf, representation='twos_complement'):
    """
    Compute bit-level sparsity of a quantized fixed-point weight tensor.

    Args:
        weight_tensor: PyTorch tensor or numpy array of (already quantized) float weights
        qi: Number of integer bits (including sign bit)
        qf: Number of fractional bits
        representation: 'twos_complement' (default) or 'sign_magnitude' (BitWave)

    Returns:
        dict with element-level and bit-level sparsity statistics
    """
    bw = qi + qf  # Total bit width

    # Convert to numpy
    if isinstance(weight_tensor, torch.Tensor):
        weights = weight_tensor.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        weights = np.asarray(weight_tensor).flatten().astype(np.float64)

    total_elements = len(weights)

    # Float -> fixed-point integer (replicating cast_fp_to_fxp logic)
    power = 2.0 ** qf
    clip_max = 2.0 ** (bw - 1) - 1
    clip_min = -(2.0 ** (bw - 1))
    int_weights = np.round(weights * power).clip(clip_min, clip_max).astype(np.int32)

    # Element-level sparsity
    zero_elements = int(np.sum(int_weights == 0))
    nonzero_elements = total_elements - zero_elements
    element_sparsity = zero_elements / total_elements if total_elements > 0 else 0.0

    # Convert to unsigned representation for bit counting
    if representation == 'sign_magnitude':
        unsigned = _to_unsigned_sign_magnitude(int_weights, bw)
    else:
        unsigned = _to_unsigned_twos_complement(int_weights, bw)

    # Count non-zero bits across ALL elements
    total_bits = total_elements * bw
    nonzero_bits_all = _count_nonzero_bits(unsigned, bw)
    zero_bits_all = total_bits - nonzero_bits_all
    bit_sparsity = zero_bits_all / total_bits if total_bits > 0 else 0.0

    # Bit-level sparsity of NON-ZERO elements only
    if nonzero_elements > 0:
        nz_mask = int_weights != 0
        unsigned_nz = unsigned[nz_mask]
        total_bits_nz = nonzero_elements * bw
        nonzero_bits_nz = _count_nonzero_bits(unsigned_nz, bw)
        zero_bits_nz = total_bits_nz - nonzero_bits_nz
        bit_sparsity_nonzero = zero_bits_nz / total_bits_nz if total_bits_nz > 0 else 0.0
        avg_nonzero_bits_per_nz_element = nonzero_bits_nz / nonzero_elements
    else:
        bit_sparsity_nonzero = 0.0
        avg_nonzero_bits_per_nz_element = 0.0
        nonzero_bits_nz = 0

    avg_nonzero_bits_per_element = nonzero_bits_all / total_elements if total_elements > 0 else 0.0

    return {
        'bit_width': bw,
        'qi': qi,
        'qf': qf,
        'representation': representation,
        'total_elements': total_elements,
        'nonzero_elements': nonzero_elements,
        'zero_elements': zero_elements,
        'element_sparsity': element_sparsity,
        'total_bits': total_bits,
        'nonzero_bits': nonzero_bits_all,
        'zero_bits': zero_bits_all,
        'bit_sparsity': bit_sparsity,
        'bit_sparsity_nonzero': bit_sparsity_nonzero,
        'avg_nonzero_bits_per_element': avg_nonzero_bits_per_element,
        'avg_nonzero_bits_per_nonzero_element': avg_nonzero_bits_per_nz_element,
    }


def analyze_sparsity_scenarios(net, args, temporal_stats=None, representation='twos_complement'):
    """
    Analyze all 5 sparsity exploitation scenarios.

    Args:
        net: PyTorch model (Model class)
        args: Argument namespace with wqi, wqf, etc.
        temporal_stats: dict with 'sp_dx', 'sp_dh', 'sp_dxh' from evaluation
        representation: 'twos_complement' or 'sign_magnitude'

    Returns:
        dict with per-layer results, aggregate results, and speedup calculations
    """
    qi = args.wqi
    qf = args.wqf
    bw = qi + qf

    # --- Per-layer analysis ---
    layer_results = []

    # Aggregate counters for RNN weights
    rnn_total_elements = 0
    rnn_nonzero_elements = 0
    rnn_total_bits = 0
    rnn_nonzero_bits = 0
    rnn_nz_total_bits = 0
    rnn_nz_nonzero_bits = 0

    for name, param in net.named_parameters():
        if 'rnn' in name and 'weight' in name:
            stats = compute_bit_level_sparsity(param.data, qi, qf, representation)
            stats['name'] = name
            stats['shape'] = list(param.shape)
            layer_results.append(stats)

            rnn_total_elements += stats['total_elements']
            rnn_nonzero_elements += stats['nonzero_elements']
            rnn_total_bits += stats['total_bits']
            rnn_nonzero_bits += stats['nonzero_bits']
            rnn_nz_total_bits += stats['nonzero_elements'] * bw
            rnn_nz_nonzero_bits += int(stats['avg_nonzero_bits_per_nonzero_element'] * stats['nonzero_elements'])

    # FC layer analysis (separate, for reference)
    fc_results = []
    for name, param in net.named_parameters():
        if ('fc_final' in name or 'fc_extra' in name) and 'weight' in name:
            stats = compute_bit_level_sparsity(param.data, qi, qf, representation)
            stats['name'] = name
            stats['shape'] = list(param.shape)
            fc_results.append(stats)

    # --- Aggregate RNN sparsity ---
    rnn_zero_elements = rnn_total_elements - rnn_nonzero_elements
    sp_w = rnn_zero_elements / rnn_total_elements if rnn_total_elements > 0 else 0.0
    bit_sp = (rnn_total_bits - rnn_nonzero_bits) / rnn_total_bits if rnn_total_bits > 0 else 0.0
    bit_sp_nz = (rnn_nz_total_bits - rnn_nz_nonzero_bits) / rnn_nz_total_bits if rnn_nz_total_bits > 0 else 0.0

    # --- Temporal sparsity ---
    sp_dx = 0.0
    sp_dh = 0.0
    sp_dxh = 0.0
    if temporal_stats is not None:
        sp_dx = temporal_stats.get('sp_dx', 0.0)
        sp_dh = temporal_stats.get('sp_dh', 0.0)
        sp_dxh = temporal_stats.get('sp_dxh', 0.0)

    # --- Compute 5 speedup scenarios ---
    eps = 1e-12  # avoid division by zero

    # Scenario 1: Weight sparsity alone
    speedup_1 = 1.0 / max(1.0 - sp_w, eps)

    # Scenario 2: Temporal sparsity alone
    speedup_2 = 1.0 / max(1.0 - sp_dxh, eps)

    # Scenario 3: Bit-level weight sparsity alone
    speedup_3 = 1.0 / max(1.0 - bit_sp, eps)

    # Scenario 4: Weight + Temporal
    speedup_4 = 1.0 / max((1.0 - sp_w) * (1.0 - sp_dxh), eps)

    # Scenario 5: Weight + Temporal + Bit-level (non-zero weights)
    speedup_5 = 1.0 / max((1.0 - sp_w) * (1.0 - sp_dxh) * (1.0 - bit_sp_nz), eps)

    scenarios = [
        {
            'id': 1,
            'name': 'Weight sparsity',
            'description': 'Skip MAC where weight = 0',
            'sparsity': sp_w,
            'speedup': speedup_1,
        },
        {
            'id': 2,
            'name': 'Temporal sparsity',
            'description': 'Skip MAC where delta = 0',
            'sparsity': sp_dxh,
            'speedup': speedup_2,
        },
        {
            'id': 3,
            'name': 'Bit-level weight sparsity',
            'description': 'Skip zero bits in bit-serial processing',
            'sparsity': bit_sp,
            'speedup': speedup_3,
        },
        {
            'id': 4,
            'name': 'Weight + Temporal',
            'description': 'Skip MAC where weight=0 OR delta=0',
            'combined_sparsity': 1.0 - (1.0 - sp_w) * (1.0 - sp_dxh),
            'speedup': speedup_4,
        },
        {
            'id': 5,
            'name': 'Weight + Temporal + Bit-level',
            'description': 'Scenario 4 + skip zero weight bits',
            'effective_reduction': 1.0 - (1.0 - sp_w) * (1.0 - sp_dxh) * (1.0 - bit_sp_nz),
            'speedup': speedup_5,
        },
    ]

    return {
        'representation': representation,
        'layer_results': layer_results,
        'fc_results': fc_results,
        'aggregate': {
            'total_elements': rnn_total_elements,
            'nonzero_elements': rnn_nonzero_elements,
            'element_sparsity': sp_w,
            'total_bits': rnn_total_bits,
            'nonzero_bits': rnn_nonzero_bits,
            'bit_sparsity': bit_sp,
            'bit_sparsity_nonzero': bit_sp_nz,
            'bit_width': bw,
            'avg_nonzero_bits_per_element': rnn_nonzero_bits / rnn_total_elements if rnn_total_elements > 0 else 0,
            'avg_nonzero_bits_per_nz_element': rnn_nz_nonzero_bits / rnn_nonzero_elements if rnn_nonzero_elements > 0 else 0,
        },
        'temporal': {
            'sp_dx': sp_dx,
            'sp_dh': sp_dh,
            'sp_dxh': sp_dxh,
        },
        'scenarios': scenarios,
    }


def print_sparsity_speedup_report(analysis_results):
    """
    Print a formatted sparsity and speedup report.
    Uses rich formatting if available, otherwise plain text.

    Args:
        analysis_results: dict returned by analyze_sparsity_scenarios()
    """
    if RICH_AVAILABLE:
        _print_rich_report(analysis_results)
    else:
        _print_plain_report(analysis_results)


def _print_rich_report(results):
    """Rich-formatted sparsity and speedup report."""
    console = Console()
    agg = results['aggregate']
    temporal = results['temporal']
    scenarios = results['scenarios']
    rep = results.get('representation', 'twos_complement')
    rep_label = "Sign-Magnitude (BitWave)" if rep == 'sign_magnitude' else "Two's Complement"

    console.print()
    console.print(Panel(
        f"[bold bright_cyan]Bit-Level Sparsity & Theoretical Speedup Analysis[/]\n"
        f"[dim]Representation: {rep_label}[/]",
        border_style="bright_cyan"))

    # --- Per-layer bit-level sparsity table ---
    layer_table = Table(
        title="RNN Weight Bit-Level Sparsity (per layer)",
        box=box.ROUNDED, show_header=True, header_style="bold bright_cyan"
    )
    layer_table.add_column("Layer", style="bold", width=30)
    layer_table.add_column("Shape", justify="center", width=20)
    layer_table.add_column("Elem Sp", justify="right", width=10)
    layer_table.add_column("Bit Sp (all)", justify="right", width=12)
    layer_table.add_column("Bit Sp (nz)", justify="right", width=12)
    layer_table.add_column("Avg NZ bits/elem", justify="right", width=16)

    for lr in results['layer_results']:
        layer_table.add_row(
            lr['name'][:30],
            str(lr['shape']),
            f"{lr['element_sparsity']*100:.1f}%",
            f"{lr['bit_sparsity']*100:.1f}%",
            f"{lr['bit_sparsity_nonzero']*100:.1f}%",
            f"{lr['avg_nonzero_bits_per_element']:.2f} / {lr['bit_width']}",
        )

    # FC layers (if any)
    if results['fc_results']:
        layer_table.add_row("", "", "", "", "", "")
        for lr in results['fc_results']:
            layer_table.add_row(
                f"[dim]{lr['name'][:30]}[/]",
                str(lr['shape']),
                f"{lr['element_sparsity']*100:.1f}%",
                f"{lr['bit_sparsity']*100:.1f}%",
                f"{lr['bit_sparsity_nonzero']*100:.1f}%",
                f"{lr['avg_nonzero_bits_per_element']:.2f} / {lr['bit_width']}",
            )

    # Aggregate row
    layer_table.add_row("", "", "", "", "", "")
    layer_table.add_row(
        "[bold]RNN TOTAL[/]",
        f"{agg['total_elements']:,} elems",
        f"[bold]{agg['element_sparsity']*100:.1f}%[/]",
        f"[bold]{agg['bit_sparsity']*100:.1f}%[/]",
        f"[bold]{agg['bit_sparsity_nonzero']*100:.1f}%[/]",
        f"[bold]{agg['avg_nonzero_bits_per_element']:.2f} / {agg['bit_width']}[/]",
    )

    console.print(layer_table)

    # --- Temporal sparsity summary ---
    console.print()
    temp_text = Text()
    temp_text.append("Temporal Sparsity:  ", style="bold")
    temp_text.append(f"dx={temporal['sp_dx']*100:.1f}%  ", style="yellow")
    temp_text.append(f"dh={temporal['sp_dh']*100:.1f}%  ", style="yellow")
    temp_text.append(f"dxh={temporal['sp_dxh']*100:.1f}%", style="bold yellow")
    console.print(temp_text)

    # --- Speedup scenarios table ---
    console.print()
    sp_table = Table(
        title="Theoretical Speedup Scenarios (RNN layers)",
        box=box.DOUBLE_EDGE, show_header=True, header_style="bold bright_green"
    )
    sp_table.add_column("#", justify="center", width=3)
    sp_table.add_column("Scenario", style="bold", width=32)
    sp_table.add_column("What is Skipped", width=38)
    sp_table.add_column("Speedup", justify="right", width=10)

    for s in scenarios:
        # Color the speedup value
        spd = s['speedup']
        if spd >= 10:
            color = "bold bright_green"
        elif spd >= 5:
            color = "bold green"
        elif spd >= 2:
            color = "yellow"
        else:
            color = "white"

        sp_table.add_row(
            str(s['id']),
            s['name'],
            s['description'],
            f"[{color}]{spd:.2f}x[/]",
        )

    console.print(sp_table)

    # --- Visual bar chart ---
    console.print()
    console.print("[bold]Speedup Comparison:[/]")
    max_speedup = max(s['speedup'] for s in scenarios)
    bar_width = 50

    for s in scenarios:
        spd = s['speedup']
        filled = int((spd / max_speedup) * bar_width) if max_speedup > 0 else 0
        filled = max(1, filled)  # At least 1 character

        if s['id'] <= 3:
            color = "cyan"
        elif s['id'] == 4:
            color = "green"
        else:
            color = "bright_green"

        label = f"  {s['id']}. {s['name']:<32}"
        bar = f"[{color}]{'█' * filled}[/]{'░' * (bar_width - filled)}"
        console.print(f"{label} {bar} {spd:.2f}x")

    console.print()


def _print_plain_report(results):
    """Plain text sparsity and speedup report (fallback when rich is unavailable)."""
    agg = results['aggregate']
    temporal = results['temporal']
    scenarios = results['scenarios']
    rep = results.get('representation', 'twos_complement')
    rep_label = "Sign-Magnitude (BitWave)" if rep == 'sign_magnitude' else "Two's Complement"

    print()
    print("=" * 80)
    print(f"  Bit-Level Sparsity & Theoretical Speedup Analysis  [{rep_label}]")
    print("=" * 80)

    # Per-layer table
    print()
    print("RNN Weight Bit-Level Sparsity (per layer):")
    print("-" * 80)
    print(f"{'Layer':<30} {'Shape':<20} {'Elem Sp':>8} {'Bit Sp':>8} {'Bit Sp(nz)':>10} {'NZ bits':>10}")
    print("-" * 80)
    for lr in results['layer_results']:
        print(f"{lr['name'][:30]:<30} {str(lr['shape']):<20} "
              f"{lr['element_sparsity']*100:>7.1f}% {lr['bit_sparsity']*100:>7.1f}% "
              f"{lr['bit_sparsity_nonzero']*100:>9.1f}% "
              f"{lr['avg_nonzero_bits_per_element']:>6.2f}/{lr['bit_width']}")

    if results['fc_results']:
        print()
        for lr in results['fc_results']:
            print(f"{lr['name'][:30]:<30} {str(lr['shape']):<20} "
                  f"{lr['element_sparsity']*100:>7.1f}% {lr['bit_sparsity']*100:>7.1f}% "
                  f"{lr['bit_sparsity_nonzero']*100:>9.1f}% "
                  f"{lr['avg_nonzero_bits_per_element']:>6.2f}/{lr['bit_width']}")

    print("-" * 80)
    print(f"{'RNN TOTAL':<30} {agg['total_elements']:>13,} elems "
          f"{agg['element_sparsity']*100:>7.1f}% {agg['bit_sparsity']*100:>7.1f}% "
          f"{agg['bit_sparsity_nonzero']*100:>9.1f}% "
          f"{agg['avg_nonzero_bits_per_element']:>6.2f}/{agg['bit_width']}")

    # Temporal
    print()
    print(f"Temporal Sparsity:  dx={temporal['sp_dx']*100:.1f}%  "
          f"dh={temporal['sp_dh']*100:.1f}%  dxh={temporal['sp_dxh']*100:.1f}%")

    # Speedup scenarios
    print()
    print("Theoretical Speedup Scenarios (RNN layers):")
    print("-" * 80)
    print(f"{'#':<3} {'Scenario':<32} {'What is Skipped':<34} {'Speedup':>8}")
    print("-" * 80)
    for s in scenarios:
        print(f"{s['id']:<3} {s['name']:<32} {s['description']:<34} {s['speedup']:>7.2f}x")
    print("-" * 80)
    print()
