"""
Model Statistics Utility using torchstat2
Provides comprehensive model analysis including FLOPs, MADDs, memory, and throughput.
Adapted for RNN models used in this codebase.
"""
import sys
import os
import io
import contextlib
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
import pandas as pd

# Add torchstat2 to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'torchstat2'))

try:
    from torchstat2 import ModelHook, StatTree, StatNode, report_format
    from torchstat2.throughput import throughput, analyze_batch_scaling
    TORCHSTAT2_AVAILABLE = True
except ImportError:
    TORCHSTAT2_AVAILABLE = False
    print("Warning: torchstat2 not available. Install from ./torchstat2/")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def get_parent_node(root_node, stat_node_name):
    """Get parent node in the stat tree."""
    node = root_node
    names = stat_node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i+1])
        child_index = node.find_child_index(node_name)
        if child_index == -1:
            return node
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_stat_tree(leaf_modules):
    """Convert leaf modules to stat tree."""
    root_node = StatNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            stat_node_name = '.'.join(names[0:i+1])
            parent_node = get_parent_node(root_node, stat_node_name)
            node = StatNode(name=stat_node_name, parent=parent_node)
            parent_node.add_child(node)
            if i == len(names) - 1:
                try:
                    input_shape = leaf_module.input_shape.numpy().tolist()
                    output_shape = leaf_module.output_shape.numpy().tolist()
                    node.input_shape = input_shape
                    node.output_shape = output_shape
                    node.parameter_quantity = leaf_module.parameter_quantity.numpy()[0]
                    node.inference_memory = leaf_module.inference_memory.numpy()[0]
                    node.MAdd = leaf_module.MAdd.numpy()[0]
                    node.Flops = leaf_module.Flops.numpy()[0]
                    node.duration = leaf_module.duration.numpy()[0]
                    node.Memory = leaf_module.Memory.numpy().tolist()
                    node.module_type = type(leaf_module.module).__name__ if hasattr(leaf_module, 'module') else ''
                except Exception:
                    pass
    return StatTree(root_node)


class RNNModelStats:
    """
    Model statistics analyzer for RNN models.
    Extends torchstat2 functionality for sequence models.
    """

    def __init__(self, model: nn.Module, seq_len: int, n_features: int,
                 batch_size: int = 1, device: str = 'cpu'):
        """
        Initialize the RNN model stats analyzer.

        Args:
            model: PyTorch model (should support input shape: seq_len, batch, features)
            seq_len: Sequence length
            n_features: Number of input features
            batch_size: Batch size for analysis
            device: Device for throughput testing
        """
        self.model = model
        self.seq_len = seq_len
        self.n_features = n_features
        self.batch_size = batch_size
        self.device = device

        # Statistics
        self.total_params = 0
        self.trainable_params = 0
        self.total_flops = 0
        self.total_madds = 0
        self.total_memory_mb = 0
        self.total_mem_read = 0
        self.total_mem_write = 0
        self.layer_stats = []

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = 0
        trainable = 0
        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        self.total_params = total
        self.trainable_params = trainable

        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }

    def estimate_rnn_flops(self) -> Dict[str, int]:
        """
        Estimate FLOPs for RNN models.
        LSTM: 4 gates * (input_size * hidden_size + hidden_size * hidden_size) * 2 * seq_len
        GRU: 3 gates * same formula
        """
        flops = 0
        madds = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.LSTM):
                # LSTM FLOPs per timestep per layer
                input_size = module.input_size
                hidden_size = module.hidden_size
                num_layers = module.num_layers

                # Layer 1: input is n_features
                flops_layer1 = 4 * (input_size * hidden_size + hidden_size * hidden_size) * 2
                # Other layers: input is hidden_size
                flops_other = 4 * (hidden_size * hidden_size + hidden_size * hidden_size) * 2

                layer_flops = flops_layer1 + (num_layers - 1) * flops_other
                total_rnn_flops = layer_flops * self.seq_len * self.batch_size
                flops += total_rnn_flops

            elif isinstance(module, nn.GRU):
                input_size = module.input_size
                hidden_size = module.hidden_size
                num_layers = module.num_layers

                flops_layer1 = 3 * (input_size * hidden_size + hidden_size * hidden_size) * 2
                flops_other = 3 * (hidden_size * hidden_size + hidden_size * hidden_size) * 2

                layer_flops = flops_layer1 + (num_layers - 1) * flops_other
                total_rnn_flops = layer_flops * self.seq_len * self.batch_size
                flops += total_rnn_flops

            elif isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features * 2 * self.batch_size
                if module.bias is not None:
                    flops += module.out_features * self.batch_size

            # Check for DeltaRNN
            elif hasattr(module, 'mode') and 'Delta' in str(getattr(module, 'mode', '')):
                # Similar to LSTM
                input_size = module.input_size
                hidden_size = module.hidden_size
                num_layers = module.num_layers

                flops_layer1 = 4 * (input_size * hidden_size + hidden_size * hidden_size) * 2
                flops_other = 4 * (hidden_size * hidden_size + hidden_size * hidden_size) * 2

                layer_flops = flops_layer1 + (num_layers - 1) * flops_other
                total_rnn_flops = layer_flops * self.seq_len * self.batch_size
                flops += total_rnn_flops

        self.total_flops = flops
        self.total_madds = flops // 2

        return {
            'flops': flops,
            'madds': flops // 2
        }

    def estimate_memory(self) -> Dict[str, float]:
        """Estimate memory usage."""
        # Parameter memory
        param_memory = self.total_params * 4  # float32 = 4 bytes

        # Activation memory (rough estimate)
        # For RNN: hidden states for each timestep
        activation_memory = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU)):
                # Hidden state: (num_layers, batch, hidden_size)
                activation_memory += module.num_layers * self.batch_size * module.hidden_size * 4
                # Output: (seq_len, batch, hidden_size)
                activation_memory += self.seq_len * self.batch_size * module.hidden_size * 4
            elif hasattr(module, 'hidden_size') and hasattr(module, 'num_layers'):
                activation_memory += module.num_layers * self.batch_size * module.hidden_size * 4
                activation_memory += self.seq_len * self.batch_size * module.hidden_size * 4

        total_memory = param_memory + activation_memory
        self.total_memory_mb = total_memory / (1024 * 1024)

        return {
            'param_memory_mb': param_memory / (1024 * 1024),
            'activation_memory_mb': activation_memory / (1024 * 1024),
            'total_memory_mb': self.total_memory_mb
        }

    def analyze(self) -> Dict[str, Any]:
        """Run full analysis."""
        params = self.count_parameters()
        flops = self.estimate_rnn_flops()
        memory = self.estimate_memory()

        return {
            **params,
            **flops,
            **memory,
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'batch_size': self.batch_size
        }

    def get_layer_breakdown(self) -> List[Dict[str, Any]]:
        """Get layer-by-layer breakdown with per-layer FLOPs and MADDs.

        Uses forward hooks and a dummy forward pass to capture per-layer
        statistics via torchstat2's compute_flops/compute_madd functions.
        Falls back to manual computation for DeltaRNN modules.
        """
        layers = []
        hook_handles = []

        def _make_hook(name):
            def hook_fn(module, inp, out):
                # Extract first tensor input
                first_inp = None
                if isinstance(inp, tuple):
                    for item in inp:
                        if isinstance(item, torch.Tensor):
                            first_inp = item
                            break
                elif isinstance(inp, torch.Tensor):
                    first_inp = inp

                if first_inp is None:
                    module._layer_flops = 0
                    module._layer_madds = 0
                    module._layer_input_shape = []
                    module._layer_output_shape = []
                    return

                # Extract output tensor
                if isinstance(out, torch.Tensor):
                    out_tensor = out
                elif isinstance(out, (tuple, list)):
                    out_tensor = None
                    for item in out:
                        if isinstance(item, torch.Tensor):
                            out_tensor = item
                            break
                else:
                    out_tensor = None

                # Shapes
                module._layer_input_shape = list(first_inp.shape)
                module._layer_output_shape = list(out_tensor.shape) if out_tensor is not None else []

                # Compute FLOPs and MADDs via torchstat2
                flops = 0
                madds = 0
                if TORCHSTAT2_AVAILABLE and out_tensor is not None:
                    try:
                        from torchstat2 import compute_flops as ts_flops, compute_madd as ts_madd
                        out_for_compute = out_tensor
                        # Suppress torchstat2 print warnings for unsupported modules
                        with contextlib.redirect_stdout(io.StringIO()):
                            flops = ts_flops(module, first_inp, out_for_compute)
                            madds = ts_madd(module, first_inp, out_for_compute)
                    except Exception:
                        pass

                # Fallback for DeltaRNN (not recognized by torchstat2)
                if flops == 0 and hasattr(module, 'mode') and 'Delta' in str(getattr(module, 'mode', '')):
                    input_size = module.input_size
                    hidden_size = module.hidden_size
                    num_layers = module.num_layers
                    seq_len = first_inp.shape[0]
                    batch_size = first_inp.shape[1] if first_inp.dim() > 1 else 1
                    gates = 4  # LSTM has 4 gates
                    flops_l0 = gates * (input_size * hidden_size + hidden_size * hidden_size) * 2
                    flops_li = gates * (hidden_size * hidden_size + hidden_size * hidden_size) * 2
                    total_flops_per_step = flops_l0 + max(0, num_layers - 1) * flops_li
                    flops = total_flops_per_step * seq_len * batch_size
                    madds = flops // 2

                module._layer_flops = flops
                module._layer_madds = madds
            return hook_fn

        # Register hooks on all leaf modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                h = module.register_forward_hook(_make_hook(name))
                hook_handles.append(h)

        # Run dummy forward pass to trigger hooks
        try:
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'cnn_frontend') and self.model.cnn_frontend:
                    cnn_h = getattr(self.model, 'cnn_input_height', 60)
                    cnn_w = getattr(self.model, 'cnn_input_width', 80)
                    dummy_input = torch.zeros(self.seq_len, self.batch_size,
                                              self.n_features, cnn_h, cnn_w)
                else:
                    dummy_input = torch.zeros(self.seq_len, self.batch_size,
                                              self.n_features)
                # Provide feature_lengths for DeltaRNN's update_stats
                feat_lens = torch.full((self.batch_size,), self.seq_len,
                                       dtype=torch.long)
                self.model(dummy_input, feature_lengths=feat_lens)
        except Exception:
            pass
        finally:
            for h in hook_handles:
                h.remove()

        # Collect results from hooked modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                params = sum(p.numel() for p in module.parameters(recurse=False))
                flops = getattr(module, '_layer_flops', 0)
                madds = getattr(module, '_layer_madds', 0)
                input_shape = getattr(module, '_layer_input_shape', [])
                output_shape = getattr(module, '_layer_output_shape', [])

                if params > 0 or flops > 0:
                    layers.append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'params': params,
                        'flops': flops,
                        'madds': madds,
                        'input_shape': input_shape,
                        'output_shape': output_shape,
                    })

                # Clean up temporary attributes
                for attr in ('_layer_flops', '_layer_madds',
                             '_layer_input_shape', '_layer_output_shape'):
                    if hasattr(module, attr):
                        delattr(module, attr)

        return layers


def format_number(num: int) -> str:
    """Format large numbers with K, M, G suffixes."""
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(int(num))


def print_model_stats_rich(stats: Dict[str, Any], model_name: str = "Model",
                           layer_breakdown: List[Dict] = None, args=None):
    """Print model statistics using rich formatting."""
    if not RICH_AVAILABLE:
        print_model_stats_plain(stats, model_name, layer_breakdown, args)
        return

    # Header
    console.print()
    console.print(Panel(
        f"[bold cyan]Model Analysis: {model_name}[/]",
        border_style="cyan"
    ))

    # Main statistics table
    main_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta",
                       title="Model Complexity")
    main_table.add_column("Metric", style="bold", width=28)
    main_table.add_column("Value", justify="right", width=18)
    main_table.add_column("Description", style="dim", width=30)

    main_table.add_row(
        "Total Parameters",
        f"{stats['total']:,}",
        "Weights + biases"
    )
    main_table.add_row(
        "Trainable Parameters",
        f"{stats['trainable']:,}",
        "Updated during training"
    )
    main_table.add_row(
        "Non-trainable Parameters",
        f"{stats['non_trainable']:,}",
        "Frozen parameters"
    )
    main_table.add_row("", "", "")
    main_table.add_row(
        "Total FLOPs",
        format_number(stats['flops']),
        "Floating-point ops/forward"
    )
    main_table.add_row(
        "Total MADDs",
        format_number(stats['madds']),
        "Multiply-accumulate ops"
    )
    main_table.add_row("", "", "")
    main_table.add_row(
        "Parameter Memory",
        f"{stats['param_memory_mb']:.2f} MB",
        "Weight storage"
    )
    main_table.add_row(
        "Activation Memory",
        f"{stats['activation_memory_mb']:.2f} MB",
        "Intermediate activations"
    )
    main_table.add_row(
        "Total Memory",
        f"[bold]{stats['total_memory_mb']:.2f} MB[/]",
        "Total inference memory"
    )

    console.print(main_table)

    # Input configuration
    config_table = Table(box=box.SIMPLE, show_header=False, title="Input Configuration")
    config_table.add_column("Property", style="cyan")
    config_table.add_column("Value")

    config_table.add_row("Sequence Length", str(stats['seq_len']))
    config_table.add_row("Input Features", str(stats['n_features']))
    config_table.add_row("Batch Size", str(stats['batch_size']))

    console.print(config_table)

    # Layer breakdown
    if layer_breakdown:
        has_flops = any(l.get('flops', 0) > 0 for l in layer_breakdown)
        layer_table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow",
                           title="Layer Breakdown")
        layer_table.add_column("Layer", width=30)
        layer_table.add_column("Type", width=18)
        layer_table.add_column("Params", justify="right", width=16)
        if has_flops:
            layer_table.add_column("FLOPs", justify="right", width=16)
            layer_table.add_column("MADDs", justify="right", width=16)

        total_params = sum(l['params'] for l in layer_breakdown)
        total_flops = sum(l.get('flops', 0) for l in layer_breakdown)
        total_madds = sum(l.get('madds', 0) for l in layer_breakdown)

        for layer in layer_breakdown:
            pct_p = (layer['params'] / total_params * 100) if total_params > 0 else 0
            row = [
                layer['name'][:30],
                layer['type'],
                f"{format_number(layer['params'])} ({pct_p:.1f}%)",
            ]
            if has_flops:
                lf = layer.get('flops', 0)
                pct_f = (lf / total_flops * 100) if total_flops > 0 else 0
                row.append(f"{format_number(lf)} ({pct_f:.1f}%)")
                row.append(format_number(layer.get('madds', 0)))
            layer_table.add_row(*row)

        # Total row
        total_row = [
            "[bold]Total[/]", "",
            f"[bold]{format_number(total_params)}[/]",
        ]
        if has_flops:
            total_row.append(f"[bold]{format_number(total_flops)}[/]")
            total_row.append(f"[bold]{format_number(total_madds)}[/]")
        layer_table.add_row(*total_row, end_section=True)

        console.print(layer_table)

    # DeltaLSTM specific info
    if args and hasattr(args, 'hid_type') and 'Delta' in args.hid_type:
        delta_table = Table(box=box.ROUNDED, show_header=False, title="DeltaLSTM Configuration")
        delta_table.add_column("Property", style="bold orange1")
        delta_table.add_column("Value")

        delta_table.add_row("Input Threshold (th_x)", f"{args.th_x:.3f}")
        delta_table.add_row("Hidden Threshold (th_h)", f"{args.th_h:.3f}")
        delta_table.add_row("Target Sparsity (γ_rnn)", f"{args.gamma_rnn*100:.1f}%")
        if hasattr(args, 'balance_pe'):
            delta_table.add_row("Balance PE", "Enabled" if args.balance_pe else "Disabled")
            if args.balance_pe:
                delta_table.add_row("Num Arrays", str(args.num_array))
                delta_table.add_row("PEs per Array", str(args.num_array_pe))

        console.print(delta_table)


def print_model_stats_plain(stats: Dict[str, Any], model_name: str = "Model",
                            layer_breakdown: List[Dict] = None, args=None):
    """Print model statistics in plain text."""
    print("=" * 70)
    print(f"MODEL ANALYSIS: {model_name}")
    print("=" * 70)
    print(f"Sequence Length: {stats['seq_len']}")
    print(f"Input Features:  {stats['n_features']}")
    print(f"Batch Size:      {stats['batch_size']}")
    print("-" * 70)
    print(f"Total Parameters:        {stats['total']:,}")
    print(f"Trainable Parameters:    {stats['trainable']:,}")
    print(f"Non-trainable:           {stats['non_trainable']:,}")
    print("-" * 70)
    print(f"Total FLOPs:             {format_number(stats['flops'])}")
    print(f"Total MADDs:             {format_number(stats['madds'])}")
    print("-" * 70)
    print(f"Parameter Memory:        {stats['param_memory_mb']:.2f} MB")
    print(f"Activation Memory:       {stats['activation_memory_mb']:.2f} MB")
    print(f"Total Memory:            {stats['total_memory_mb']:.2f} MB")
    print("=" * 70)

    if layer_breakdown:
        has_flops = any(l.get('flops', 0) > 0 for l in layer_breakdown)
        print("\nLAYER BREAKDOWN:")
        print("-" * (90 if has_flops else 70))
        total_params = sum(l['params'] for l in layer_breakdown)
        total_flops = sum(l.get('flops', 0) for l in layer_breakdown)
        total_madds = sum(l.get('madds', 0) for l in layer_breakdown)
        if has_flops:
            print(f"  {'Layer':<30} {'Type':<15} {'Params':>12}  {'FLOPs':>12}  {'MADDs':>12}")
            print(f"  {'-'*30} {'-'*15} {'-'*12}  {'-'*12}  {'-'*12}")
        for layer in layer_breakdown:
            pct_p = (layer['params'] / total_params * 100) if total_params > 0 else 0
            line = f"  {layer['name'][:30]:<30} {layer['type']:<15} {format_number(layer['params']):>10} ({pct_p:.1f}%)"
            if has_flops:
                lf = layer.get('flops', 0)
                pct_f = (lf / total_flops * 100) if total_flops > 0 else 0
                line += f"  {format_number(lf):>10} ({pct_f:.1f}%)"
                line += f"  {format_number(layer.get('madds', 0)):>10}"
            print(line)
        print("-" * (90 if has_flops else 70))
        line = f"  {'Total':<30} {'':<15} {format_number(total_params):>10}"
        if has_flops:
            line += f"         {format_number(total_flops):>10}         {format_number(total_madds):>10}"
        print(line)


def analyze_rnn_model(model: nn.Module, seq_len: int, n_features: int,
                      batch_size: int = 1, device: str = 'cpu',
                      model_name: str = "Model", args=None,
                      print_stats: bool = True) -> Dict[str, Any]:
    """
    Analyze an RNN model and optionally print statistics.

    Args:
        model: PyTorch model
        seq_len: Sequence length
        n_features: Number of input features
        batch_size: Batch size for analysis
        device: Device for throughput testing
        model_name: Name for display
        args: Arguments namespace (for DeltaLSTM config)
        print_stats: Whether to print statistics

    Returns:
        Dictionary with all statistics
    """
    analyzer = RNNModelStats(model, seq_len, n_features, batch_size, device)
    stats = analyzer.analyze()
    layer_breakdown = analyzer.get_layer_breakdown()

    # Use layer breakdown totals (more accurate — includes CNN + RNN + FC)
    if layer_breakdown:
        lb_flops = sum(l.get('flops', 0) for l in layer_breakdown)
        lb_madds = sum(l.get('madds', 0) for l in layer_breakdown)
        if lb_flops > 0:
            stats['flops'] = lb_flops
            stats['madds'] = lb_madds

    if print_stats:
        print_model_stats_rich(stats, model_name, layer_breakdown, args)

    stats['layer_breakdown'] = layer_breakdown
    return stats


def run_torchstat2_analysis(model: nn.Module, input_size: Tuple[int, ...],
                            print_report: bool = True) -> Optional[str]:
    """
    Run torchstat2 analysis on a model.

    Args:
        model: PyTorch model (will be moved to CPU for analysis)
        input_size: Input size tuple (e.g., (3, 224, 224) for images)
        print_report: Whether to print the report

    Returns:
        Report string or None if torchstat2 not available
    """
    if not TORCHSTAT2_AVAILABLE:
        print("torchstat2 not available")
        return None

    # Move model to CPU for torchstat2
    model_cpu = model.cpu()
    model_cpu.eval()

    try:
        model_hook = ModelHook(model_cpu, input_size)
        leaf_modules = model_hook.retrieve_leaf_modules()
        stat_tree = convert_leaf_modules_to_stat_tree(leaf_modules)
        collected_nodes = stat_tree.get_collected_stat_nodes(1)
        report = report_format(collected_nodes)

        if print_report:
            print(report)

        return report

    except Exception as e:
        print(f"torchstat2 analysis failed: {e}")
        return None


def run_throughput_analysis(model: nn.Module, input_shape: Tuple[int, ...],
                            device: str = 'cuda',
                            batch_sizes: List[int] = [1, 8, 16, 32],
                            model_name: str = "Model") -> Dict[str, Any]:
    """
    Run throughput analysis using torchstat2.

    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        device: Device to run on
        batch_sizes: Batch sizes to test
        model_name: Name for display

    Returns:
        Throughput results dictionary
    """
    if not TORCHSTAT2_AVAILABLE:
        print("torchstat2 not available for throughput analysis")
        return {}

    return throughput(
        model=model,
        input_shape=input_shape,
        device=device,
        batch_sizes=batch_sizes,
        model_name=model_name
    )


def get_model_summary_dict(model: nn.Module, seq_len: int, n_features: int,
                           batch_size: int = 1) -> Dict[str, Any]:
    """
    Get a dictionary summary of model statistics for logging.

    Args:
        model: PyTorch model
        seq_len: Sequence length
        n_features: Number of input features
        batch_size: Batch size

    Returns:
        Dictionary with statistics suitable for logging
    """
    analyzer = RNNModelStats(model, seq_len, n_features, batch_size)
    stats = analyzer.analyze()
    layer_breakdown = analyzer.get_layer_breakdown()

    # Use layer breakdown totals when available (includes CNN + RNN + FC)
    flops = stats['flops']
    madds = stats['madds']
    if layer_breakdown:
        lb_flops = sum(l.get('flops', 0) for l in layer_breakdown)
        lb_madds = sum(l.get('madds', 0) for l in layer_breakdown)
        if lb_flops > 0:
            flops = lb_flops
            madds = lb_madds

    return {
        'n_params_total': stats['total'],
        'n_params_trainable': stats['trainable'],
        'flops': flops,
        'madds': madds,
        'memory_params_mb': stats['param_memory_mb'],
        'memory_activations_mb': stats['activation_memory_mb'],
        'memory_total_mb': stats['total_memory_mb'],
        'seq_len': seq_len,
        'n_features': n_features,
        'batch_size': batch_size,
        'layer_breakdown': [
            {
                'name': l['name'],
                'type': l['type'],
                'params': l['params'],
                'flops': l.get('flops', 0),
                'madds': l.get('madds', 0),
            }
            for l in layer_breakdown
        ]
    }
