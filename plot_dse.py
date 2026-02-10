#!/usr/bin/env python3
"""
plot_dse.py — Design Space Exploration Plotter for DeltaLSTM-CBTD

Reads merged log CSVs produced by merge_logs.py and generates publication-quality
plots analyzing weight sparsity, temporal sparsity, bit sparsity, accuracy, and
theoretical speedup across the design space.

Usage:
    python plot_dse.py --dataset_name gscdv2
    python plot_dse.py --dataset_name gscdv2 --output_dir ./plot/dse
    python plot_dse.py --retrain_csv ./log/gscdv2/retrain/merged/merged_test_*.csv
"""

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter, FuncFormatter

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Try loading torch + sparsity analysis for bit-level computation
try:
    import torch
    from utils.sparsity_analysis import compute_bit_level_sparsity
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# Style configuration
# =============================================================================
SAVE_DPI = 300
FIG_FORMATS = ['png', 'pdf']

# Professional color palette
PALETTE = {
    'blue':    '#4472C4',
    'orange':  '#ED7D31',
    'green':   '#70AD47',
    'red':     '#C00000',
    'purple':  '#7030A0',
    'teal':    '#00B0F0',
    'gray':    '#A5A5A5',
}

SCENARIO_COLORS = {
    'Weight':              PALETTE['blue'],
    'Temporal':            PALETTE['orange'],
    'Bit-level':           PALETTE['green'],
    'Weight+Temporal':     PALETTE['red'],
    'Weight+Temporal+Bit': PALETTE['purple'],
}


def setup_style():
    """Configure matplotlib / seaborn for publication-quality plots."""
    if HAS_SEABORN:
        sns.set_theme(style='whitegrid', font_scale=1.1,
                      rc={'axes.edgecolor': '.3', 'grid.color': '.9'})
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': SAVE_DPI,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 15,
    })


# =============================================================================
# Data loading
# =============================================================================

def find_latest_merged(log_dir, dataset, phase, source):
    """Return the most recent merged CSV path, or None."""
    pattern = os.path.join(log_dir, dataset, phase, 'merged', f'merged_{source}_*.csv')
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def load_and_prepare(csv_path):
    """Load a merged CSV, add clean aliases and computed columns."""
    df = pd.read_csv(csv_path)

    # Clean aliases for swept hyperparameters
    df['gamma'] = df['HP_SPRNN'].astype(float) if 'HP_SPRNN' in df.columns else 0.0
    if 'HP_TX' in df.columns:
        df['threshold'] = df['HP_TX'].astype(float)
    else:
        df['threshold'] = 0.0
    df['msq_lambda'] = df['HP_MSQL'].fillna(0).astype(float) if 'HP_MSQL' in df.columns else 0.0

    # Prefer TEST sparsity, fall back to VAL
    for sp in ['SP_DX', 'SP_DH', 'SP_DXH']:
        test_col = f'TEST_{sp}'
        val_col  = f'VAL_{sp}'
        col_name = sp.lower()
        if test_col in df.columns:
            df[col_name] = pd.to_numeric(df[test_col], errors='coerce').fillna(0)
        elif val_col in df.columns:
            df[col_name] = pd.to_numeric(df[val_col], errors='coerce').fillna(0)
        else:
            df[col_name] = 0.0

    df['sp_w'] = pd.to_numeric(df['SP_W_RNN'], errors='coerce').fillna(0) if 'SP_W_RNN' in df.columns else 0.0

    # Bit-level sparsity from CSV (two's complement)
    if 'SP_BIT_RNN' in df.columns:
        df['bit_sp'] = pd.to_numeric(df['SP_BIT_RNN'], errors='coerce').fillna(0)
    if 'SP_BIT_NZ_RNN' in df.columns:
        df['bit_sp_nz'] = pd.to_numeric(df['SP_BIT_NZ_RNN'], errors='coerce').fillna(0)
    # Bit-level sparsity from CSV (sign-magnitude / BitWave)
    if 'SP_BIT_SM_RNN' in df.columns:
        df['bit_sp_sm'] = pd.to_numeric(df['SP_BIT_SM_RNN'], errors='coerce').fillna(0)
    if 'SP_BIT_NZ_SM_RNN' in df.columns:
        df['bit_sp_nz_sm'] = pd.to_numeric(df['SP_BIT_NZ_SM_RNN'], errors='coerce').fillna(0)

    # Accuracy
    for col in ['TEST_ACCURACY', 'VAL_ACCURACY', 'TEST_F1_MACRO', 'VAL_F1_MACRO']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def compute_speedup(df):
    """Add speedup columns (scenarios 1, 2, 4; and 3, 5 if bit sparsity available)."""
    eps = 1e-6
    sp_w   = df['sp_w'].clip(upper=1 - eps)
    sp_dxh = df['sp_dxh'].clip(upper=1 - eps)

    df['speedup_weight']   = 1.0 / (1.0 - sp_w)                      # Scenario 1
    df['speedup_temporal'] = 1.0 / (1.0 - sp_dxh)                    # Scenario 2
    df['speedup_wt']       = 1.0 / ((1.0 - sp_w) * (1.0 - sp_dxh))  # Scenario 4

    # Combined sparsity: probability that a MAC is needed
    df['combined_sparsity'] = 1.0 - (1.0 - sp_w) * (1.0 - sp_dxh)

    if 'bit_sp' in df.columns:
        bit_sp    = df['bit_sp'].clip(upper=1 - eps)
        bit_sp_nz = df['bit_sp_nz'].clip(upper=1 - eps)
        df['speedup_bit']  = 1.0 / (1.0 - bit_sp)                                        # Scenario 3
        df['speedup_wtb']  = 1.0 / ((1.0 - sp_w) * (1.0 - sp_dxh) * (1.0 - bit_sp_nz))  # Scenario 5
    return df


def try_add_bit_sparsity(df, dataset, save_dir='./save'):
    """Attempt to compute bit-level sparsity from saved model weights."""
    if not HAS_TORCH:
        return df

    bit_sp_list     = []
    bit_sp_nz_list  = []
    bit_sp_sm_list  = []       # sign-magnitude
    bit_sp_nz_sm_list = []

    for _, row in df.iterrows():
        model_id = row.get('MODEL_ID', '')
        phase = 'retrain' if 'DeltaLSTM' in str(row.get('HP_CLA', '')) else 'pretrain'
        model_path = os.path.join(save_dir, dataset, phase, f'{model_id}.pt')

        if not os.path.isfile(model_path):
            bit_sp_list.append(np.nan)
            bit_sp_nz_list.append(np.nan)
            bit_sp_sm_list.append(np.nan)
            bit_sp_nz_sm_list.append(np.nan)
            continue

        try:
            state = torch.load(model_path, map_location='cpu', weights_only=True)
            # Collect RNN weights
            weights = []
            for name, param in state.items():
                if 'rnn' in name.lower() and 'weight' in name.lower():
                    weights.append(param.numpy())

            if not weights:
                raise ValueError("No RNN weights found")

            all_w = np.concatenate([w.flatten() for w in weights])

            # Two's complement (qi=1, qf=7 for GSCDv2)
            qi, qf = int(row.get('HP_WI', 1)), int(row.get('HP_WF', 7))
            tc = compute_bit_level_sparsity(all_w, qi, qf, 'twos_complement')
            sm = compute_bit_level_sparsity(all_w, qi, qf, 'sign_magnitude')

            # All elements
            bit_sp_list.append(tc['bit_sparsity'])
            bit_sp_sm_list.append(sm['bit_sparsity'])
            # Non-zero elements only
            bit_sp_nz_list.append(tc['bit_sparsity_nonzero'])
            bit_sp_nz_sm_list.append(sm['bit_sparsity_nonzero'])
        except Exception:
            bit_sp_list.append(np.nan)
            bit_sp_nz_list.append(np.nan)
            bit_sp_sm_list.append(np.nan)
            bit_sp_nz_sm_list.append(np.nan)

    if any(not np.isnan(v) for v in bit_sp_list):
        df['bit_sp']        = bit_sp_list
        df['bit_sp_nz']     = bit_sp_nz_list
        df['bit_sp_sm']     = bit_sp_sm_list
        df['bit_sp_nz_sm']  = bit_sp_nz_sm_list

    return df


# =============================================================================
# Helpers
# =============================================================================

def savefig(fig, output_dir, name):
    """Save figure in all configured formats."""
    for fmt in FIG_FORMATS:
        path = os.path.join(output_dir, f'{name}.{fmt}')
        fig.savefig(path, format=fmt)
    plt.close(fig)
    print(f"  Saved: {name}")


def pct_fmt(x, _=None):
    return f'{x * 100:.1f}%'


def make_pivot(df, value_col, aggfunc='mean'):
    """Build gamma × threshold pivot table."""
    return df.pivot_table(index='gamma', columns='threshold',
                          values=value_col, aggfunc=aggfunc).sort_index(
                              ascending=False)


def _annotated_heatmap(ax, pivot, fmt='.1f', cmap='RdYlGn', vmin=None, vmax=None,
                       pct=False, cbar_label=''):
    """Draw a seaborn-style annotated heatmap."""
    display = pivot.copy()
    if pct:
        display = display * 100
        if cbar_label and '%' not in cbar_label:
            cbar_label += ' (%)'
    if HAS_SEABORN:
        sns.heatmap(display, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.8, linecolor='white',
                    cbar_kws={'label': cbar_label, 'shrink': 0.8},
                    ax=ax)
    else:
        im = ax.imshow(display.values, cmap=cmap, aspect='auto',
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(display.columns)))
        ax.set_xticklabels(display.columns)
        ax.set_yticks(range(len(display.index)))
        ax.set_yticklabels(display.index)
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
        for i in range(len(display.index)):
            for j in range(len(display.columns)):
                v = display.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:{fmt}}', ha='center', va='center', fontsize=9)
    ax.set_xlabel('Delta Threshold (th)')
    ax.set_ylabel('Target Weight Sparsity (γ)')


# =============================================================================
# Plot functions — Category 1: Accuracy Landscape
# =============================================================================

def plot_accuracy_heatmaps(df, output_dir):
    """Heatmaps of test accuracy and F1 across the design space."""
    msq_groups = df.groupby('msq_lambda') if df['msq_lambda'].nunique() > 1 else [(0.0, df)]

    for msq_val, grp in msq_groups:
        suffix = f'_msq{msq_val}' if df['msq_lambda'].nunique() > 1 else ''

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(f'Accuracy Landscape{" (MSQ λ=" + str(msq_val) + ")" if suffix else ""}',
                     fontweight='bold')

        # Test accuracy
        pivot_acc = make_pivot(grp, 'TEST_ACCURACY')
        _annotated_heatmap(axes[0], pivot_acc, fmt='.2f', cmap='RdYlGn',
                           pct=True, cbar_label='Test Accuracy')
        axes[0].set_title('Test Accuracy')

        # F1 macro
        if 'TEST_F1_MACRO' in grp.columns and grp['TEST_F1_MACRO'].notna().any():
            pivot_f1 = make_pivot(grp, 'TEST_F1_MACRO')
            _annotated_heatmap(axes[1], pivot_f1, fmt='.2f', cmap='RdYlGn',
                               pct=True, cbar_label='Test F1 Macro')
            axes[1].set_title('Test F1 Macro')
        else:
            axes[1].set_visible(False)

        fig.tight_layout()
        savefig(fig, output_dir, f'accuracy_heatmap{suffix}')


def plot_accuracy_lines(df, output_dir):
    """Line plot of test accuracy vs gamma, one line per threshold."""
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ['o', 's', 'D', '^', 'v', 'P']
    thresholds = sorted(df['threshold'].unique())

    for i, th in enumerate(thresholds):
        sub = df[df['threshold'] == th].sort_values('gamma')
        ax.plot(sub['gamma'], sub['TEST_ACCURACY'] * 100,
                marker=markers[i % len(markers)], markersize=7, linewidth=2,
                label=f'th = {th}')

    ax.set_xlabel('Target Weight Sparsity (γ)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy vs Weight Sparsity Target', fontweight='bold')
    ax.legend(title='Delta Threshold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, output_dir, 'accuracy_vs_gamma')


# =============================================================================
# Category 2: Sparsity Decomposition
# =============================================================================

def plot_sparsity_heatmaps(df, output_dir):
    """Heatmaps for weight sparsity, temporal sparsity, and combined."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle('Sparsity Decomposition', fontweight='bold')

    # Weight sparsity
    pivot_w = make_pivot(df, 'sp_w')
    _annotated_heatmap(axes[0], pivot_w, fmt='.1f', cmap='Blues',
                       pct=True, cbar_label='Weight Sparsity')
    axes[0].set_title('Weight Sparsity (CBTD)')

    # Temporal sparsity
    pivot_t = make_pivot(df, 'sp_dxh')
    _annotated_heatmap(axes[1], pivot_t, fmt='.1f', cmap='Oranges',
                       pct=True, cbar_label='Temporal Sparsity')
    axes[1].set_title('Temporal Sparsity (Δxh)')

    # Combined
    pivot_c = make_pivot(df, 'combined_sparsity')
    _annotated_heatmap(axes[2], pivot_c, fmt='.1f', cmap='Reds',
                       pct=True, cbar_label='Combined Sparsity')
    axes[2].set_title('Combined (Weight × Temporal)')

    fig.tight_layout()
    savefig(fig, output_dir, 'sparsity_heatmaps')


def plot_sparsity_breakdown_bars(df, output_dir):
    """Grouped bar chart showing sparsity decomposition per configuration."""
    df_sorted = df.sort_values(['gamma', 'threshold'])
    labels = [f'γ={r.gamma:.2f}\nth={r.threshold:.2f}' for _, r in df_sorted.iterrows()]

    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))

    bars_w = ax.bar(x - width, df_sorted['sp_w'] * 100, width,
                    label='Weight (CBTD)', color=PALETTE['blue'], edgecolor='white')
    bars_t = ax.bar(x, df_sorted['sp_dxh'] * 100, width,
                    label='Temporal (Δxh)', color=PALETTE['orange'], edgecolor='white')
    bars_c = ax.bar(x + width, df_sorted['combined_sparsity'] * 100, width,
                    label='Combined', color=PALETTE['red'], edgecolor='white')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Sparsity Decomposition per Configuration', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    savefig(fig, output_dir, 'sparsity_breakdown_bars')


def plot_temporal_sparsity_lines(df, output_dir):
    """Temporal sparsity components (Δx, Δh, Δxh) vs threshold per gamma."""
    gammas = sorted(df['gamma'].unique())
    n = len(gammas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    fig.suptitle('Temporal Sparsity Components', fontweight='bold')

    for i, gamma in enumerate(gammas):
        ax = axes[0, i]
        sub = df[df['gamma'] == gamma].sort_values('threshold')
        ax.plot(sub['threshold'], sub['sp_dx'] * 100, 'o-', label='Δx (input)',
                color=PALETTE['blue'], linewidth=2)
        ax.plot(sub['threshold'], sub['sp_dh'] * 100, 's-', label='Δh (hidden)',
                color=PALETTE['orange'], linewidth=2)
        ax.plot(sub['threshold'], sub['sp_dxh'] * 100, 'D-', label='Δxh (combined)',
                color=PALETTE['red'], linewidth=2)
        ax.set_xlabel('Delta Threshold (th)')
        if i == 0:
            ax.set_ylabel('Temporal Sparsity (%)')
        ax.set_title(f'γ = {gamma}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    fig.tight_layout()
    savefig(fig, output_dir, 'temporal_sparsity_lines')


# =============================================================================
# Category 3: Speedup Analysis
# =============================================================================

def plot_speedup_heatmap(df, output_dir):
    """Heatmap of combined speedup (Scenario 4: Weight + Temporal)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Theoretical Speedup', fontweight='bold')

    pivot_su = make_pivot(df, 'speedup_wt')
    _annotated_heatmap(axes[0], pivot_su, fmt='.1f', cmap='YlOrRd',
                       cbar_label='Speedup (×)')
    axes[0].set_title('Weight + Temporal Speedup (×)')

    # Also show as FLOPs reduction percentage
    pivot_red = make_pivot(df, 'combined_sparsity')
    _annotated_heatmap(axes[1], pivot_red, fmt='.1f', cmap='YlOrRd',
                       pct=True, cbar_label='FLOPs Reduction')
    axes[1].set_title('FLOPs Reduction (%)')

    fig.tight_layout()
    savefig(fig, output_dir, 'speedup_heatmap')


def plot_speedup_scenarios_bar(df, output_dir):
    """Grouped bar chart showing all available speedup scenarios per config."""
    df_sorted = df.sort_values(['gamma', 'threshold'])
    labels = [f'γ={r.gamma:.2f}\nth={r.threshold:.2f}' for _, r in df_sorted.iterrows()]

    scenarios = [
        ('speedup_weight',   'Weight',              SCENARIO_COLORS['Weight']),
        ('speedup_temporal', 'Temporal',             SCENARIO_COLORS['Temporal']),
        ('speedup_wt',       'Weight+Temporal',      SCENARIO_COLORS['Weight+Temporal']),
    ]
    if 'speedup_bit' in df.columns:
        scenarios.insert(2, ('speedup_bit', 'Bit-level', SCENARIO_COLORS['Bit-level']))
        scenarios.append(('speedup_wtb', 'Weight+Temporal+Bit', SCENARIO_COLORS['Weight+Temporal+Bit']))

    n_bars = len(scenarios)
    x = np.arange(len(labels))
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 1.5), 6.5))

    for i, (col, label, color) in enumerate(scenarios):
        offset = (i - n_bars / 2 + 0.5) * width
        vals = df_sorted[col].values
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor='white')
        # Annotate bars with values
        for bar, val in zip(bars, vals):
            if val > 1.1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.1f}×', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Speedup (×)')
    ax.set_title('Theoretical Speedup — All Scenarios', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(title='Scenario', loc='upper left')
    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    savefig(fig, output_dir, 'speedup_scenarios_bar')


# =============================================================================
# Category 4: Accuracy–Speedup Trade-off
# =============================================================================

def plot_accuracy_vs_speedup(df, output_dir):
    """Scatter plot of accuracy vs combined speedup with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(9, 7))

    gammas = sorted(df['gamma'].unique())
    cmap = plt.cm.get_cmap('viridis', max(len(gammas), 2))

    for i, gamma in enumerate(gammas):
        sub = df[df['gamma'] == gamma]
        sc = ax.scatter(sub['speedup_wt'], sub['TEST_ACCURACY'] * 100,
                        c=[cmap(i)] * len(sub), s=80 + sub['threshold'] * 200,
                        marker='o', edgecolors='white', linewidth=0.8,
                        label=f'γ = {gamma}', zorder=3)

    # Pareto frontier (maximize both speedup and accuracy)
    su = df['speedup_wt'].values
    acc = (df['TEST_ACCURACY'] * 100).values
    pareto_idx = _pareto_frontier_indices(su, acc)
    if len(pareto_idx) > 1:
        pareto_pts = sorted(pareto_idx, key=lambda i: su[i])
        ax.plot(su[pareto_pts], acc[pareto_pts], 'k--', linewidth=1.5,
                alpha=0.6, label='Pareto frontier', zorder=2)

    ax.set_xlabel('Combined Speedup (Weight + Temporal) (×)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy vs Speedup Trade-off', fontweight='bold')
    ax.legend(title='Target Weight Sparsity')
    ax.grid(True, alpha=0.3)

    # Annotate Pareto-optimal points
    for idx in pareto_idx:
        row = df.iloc[idx]
        ax.annotate(f'th={row.threshold:.2f}',
                    (su[idx], acc[idx]),
                    textcoords='offset points', xytext=(8, 5),
                    fontsize=7, alpha=0.7)

    fig.tight_layout()
    savefig(fig, output_dir, 'accuracy_vs_speedup')


def _pareto_frontier_indices(x, y):
    """Find Pareto-optimal indices (maximize both x and y)."""
    sorted_idx = sorted(range(len(x)), key=lambda i: -x[i])
    pareto = []
    best_y = -np.inf
    for i in sorted_idx:
        if y[i] > best_y:
            pareto.append(i)
            best_y = y[i]
    return pareto


# =============================================================================
# Category 5: Bit-Level Sparsity (MSQ / BitWave)
# =============================================================================

def plot_bit_sparsity(df, output_dir):
    """Bit-level sparsity comparison: two's complement vs sign-magnitude."""
    if 'bit_sp' not in df.columns:
        return

    df_sorted = df.sort_values(['gamma', 'threshold'])
    labels = [f'γ={r.gamma:.2f}\nth={r.threshold:.2f}' for _, r in df_sorted.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(labels) * 1.4), 6))
    fig.suptitle('Bit-Level Weight Sparsity', fontweight='bold')

    # All elements
    ax = axes[0]
    ax.bar(x - width / 2, df_sorted['bit_sp'] * 100, width,
           label="Two's complement", color=PALETTE['blue'], edgecolor='white')
    if 'bit_sp_sm' in df.columns:
        ax.bar(x + width / 2, df_sorted['bit_sp_sm'] * 100, width,
               label='Sign-magnitude (BitWave)', color=PALETTE['orange'], edgecolor='white')
    ax.set_title('All Weight Elements')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Bit Sparsity (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Non-zero elements only
    ax = axes[1]
    ax.bar(x - width / 2, df_sorted['bit_sp_nz'] * 100, width,
           label="Two's complement", color=PALETTE['blue'], edgecolor='white')
    if 'bit_sp_nz_sm' in df.columns:
        ax.bar(x + width / 2, df_sorted['bit_sp_nz_sm'] * 100, width,
               label='Sign-magnitude (BitWave)', color=PALETTE['orange'], edgecolor='white')
    ax.set_title('Non-Zero Weight Elements Only')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Bit Sparsity (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    savefig(fig, output_dir, 'bit_sparsity')


def plot_msq_comparison(df, output_dir):
    """Compare accuracy and bit sparsity with/without MSQ."""
    if df['msq_lambda'].nunique() < 2 or 'bit_sp' not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('MSQ Bit Sparsification Effect', fontweight='bold')

    gammas = sorted(df['gamma'].unique())
    thresholds = sorted(df['threshold'].unique())

    # Use a "no MSQ" and "with MSQ" comparison
    df_no_msq  = df[df['msq_lambda'] == 0].sort_values(['gamma', 'threshold'])
    df_msq     = df[df['msq_lambda'] > 0].sort_values(['gamma', 'threshold'])

    if df_no_msq.empty or df_msq.empty:
        plt.close(fig)
        return

    labels = [f'γ={r.gamma:.2f}, th={r.threshold:.2f}' for _, r in df_no_msq.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    # Accuracy comparison
    ax = axes[0]
    ax.bar(x - width / 2, df_no_msq['TEST_ACCURACY'].values * 100, width,
           label='No MSQ', color=PALETTE['blue'], edgecolor='white')
    ax.bar(x + width / 2, df_msq['TEST_ACCURACY'].values * 100, width,
           label=f'MSQ λ={df_msq.msq_lambda.iloc[0]}', color=PALETTE['orange'],
           edgecolor='white')
    ax.set_title('Test Accuracy')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Bit sparsity comparison
    ax = axes[1]
    ax.bar(x - width / 2, df_no_msq['bit_sp_nz'].values * 100, width,
           label='No MSQ', color=PALETTE['blue'], edgecolor='white')
    ax.bar(x + width / 2, df_msq['bit_sp_nz'].values * 100, width,
           label=f'MSQ λ={df_msq.msq_lambda.iloc[0]}', color=PALETTE['orange'],
           edgecolor='white')
    ax.set_title('Bit Sparsity (Non-Zero Elements)')
    ax.set_ylabel('Bit Sparsity (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    savefig(fig, output_dir, 'msq_comparison')


# =============================================================================
# Category 6: Summary Dashboard
# =============================================================================

def plot_summary_dashboard(df, output_dir):
    """Multi-panel summary combining the key insights."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Design Space Exploration — Summary Dashboard', fontweight='bold',
                 fontsize=16, y=0.98)

    # (0,0) Accuracy heatmap
    ax = fig.add_subplot(gs[0, 0])
    pivot_acc = make_pivot(df, 'TEST_ACCURACY')
    _annotated_heatmap(ax, pivot_acc, fmt='.2f', cmap='RdYlGn',
                       pct=True, cbar_label='Accuracy (%)')
    ax.set_title('Test Accuracy', fontweight='bold')

    # (0,1) Weight sparsity heatmap
    ax = fig.add_subplot(gs[0, 1])
    pivot_w = make_pivot(df, 'sp_w')
    _annotated_heatmap(ax, pivot_w, fmt='.1f', cmap='Blues',
                       pct=True, cbar_label='Sparsity (%)')
    ax.set_title('Weight Sparsity (CBTD)', fontweight='bold')

    # (0,2) Temporal sparsity heatmap
    ax = fig.add_subplot(gs[0, 2])
    pivot_t = make_pivot(df, 'sp_dxh')
    _annotated_heatmap(ax, pivot_t, fmt='.1f', cmap='Oranges',
                       pct=True, cbar_label='Sparsity (%)')
    ax.set_title('Temporal Sparsity (Δxh)', fontweight='bold')

    # (1,0) Combined speedup heatmap
    ax = fig.add_subplot(gs[1, 0])
    pivot_su = make_pivot(df, 'speedup_wt')
    _annotated_heatmap(ax, pivot_su, fmt='.1f', cmap='YlOrRd',
                       cbar_label='Speedup (×)')
    ax.set_title('Combined Speedup (×)', fontweight='bold')

    # (1,1) Accuracy vs speedup scatter
    ax = fig.add_subplot(gs[1, 1])
    gammas = sorted(df['gamma'].unique())
    cmap_v = plt.cm.get_cmap('viridis', max(len(gammas), 2))
    for i, gamma in enumerate(gammas):
        sub = df[df['gamma'] == gamma]
        ax.scatter(sub['speedup_wt'], sub['TEST_ACCURACY'] * 100,
                   c=[cmap_v(i)] * len(sub), s=70, marker='o',
                   edgecolors='white', linewidth=0.6, label=f'γ={gamma}')
    su = df['speedup_wt'].values
    acc = (df['TEST_ACCURACY'] * 100).values
    pareto = _pareto_frontier_indices(su, acc)
    if len(pareto) > 1:
        p = sorted(pareto, key=lambda i: su[i])
        ax.plot(su[p], acc[p], 'k--', linewidth=1.2, alpha=0.5)
    ax.set_xlabel('Speedup (×)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Speedup', fontweight='bold')
    ax.legend(fontsize=7, title='γ')
    ax.grid(True, alpha=0.3)

    # (1,2) Sparsity breakdown bars (top configs by speedup)
    ax = fig.add_subplot(gs[1, 2])
    top_n = min(8, len(df))
    df_top = df.nlargest(top_n, 'speedup_wt').sort_values('speedup_wt')
    labels_top = [f'γ={r.gamma:.1f},th={r.threshold:.2f}' for _, r in df_top.iterrows()]
    y_pos = np.arange(len(labels_top))
    ax.barh(y_pos, df_top['sp_w'] * 100, 0.35, label='Weight',
            color=PALETTE['blue'], edgecolor='white')
    ax.barh(y_pos + 0.35, df_top['sp_dxh'] * 100, 0.35, label='Temporal',
            color=PALETTE['orange'], edgecolor='white')
    ax.set_yticks(y_pos + 0.175)
    ax.set_yticklabels(labels_top, fontsize=7)
    ax.set_xlabel('Sparsity (%)')
    ax.set_title('Top Configs by Speedup', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(axis='x', alpha=0.3)

    savefig(fig, output_dir, 'summary_dashboard')


# =============================================================================
# Category 7: Pretrain Comparison (LSTM baseline)
# =============================================================================

def plot_pretrain_vs_retrain(df_retrain, df_pretrain, output_dir):
    """Compare LSTM pretrain accuracy with DeltaLSTM retrain accuracy."""
    if df_pretrain is None or df_pretrain.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('LSTM Pretrain vs DeltaLSTM Retrain', fontweight='bold')

    gammas = sorted(df_retrain['gamma'].unique())
    markers = ['o', 's', 'D', '^', 'v', 'P']

    # Accuracy comparison
    ax = axes[0]
    for i, gamma in enumerate(gammas):
        sub_r = df_retrain[df_retrain['gamma'] == gamma].sort_values('threshold')
        ax.plot(sub_r['threshold'], sub_r['TEST_ACCURACY'] * 100,
                marker=markers[i % len(markers)], linewidth=2,
                label=f'DeltaLSTM γ={gamma}')
        # LSTM baseline (single point per gamma, no threshold dependence)
        sub_p = df_pretrain[df_pretrain['gamma'] == gamma]
        if not sub_p.empty:
            baseline_acc = sub_p['TEST_ACCURACY'].values[0] * 100
            ax.axhline(baseline_acc, color=f'C{i}', linestyle=':', alpha=0.5)
            ax.annotate(f'LSTM γ={gamma}: {baseline_acc:.1f}%',
                        xy=(ax.get_xlim()[1], baseline_acc),
                        fontsize=7, alpha=0.6, ha='right', va='bottom')

    ax.set_xlabel('Delta Threshold (th)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Accuracy drop from pretrain baseline
    ax = axes[1]
    for i, gamma in enumerate(gammas):
        sub_r = df_retrain[df_retrain['gamma'] == gamma].sort_values('threshold')
        sub_p = df_pretrain[df_pretrain['gamma'] == gamma]
        if sub_p.empty:
            continue
        baseline = sub_p['TEST_ACCURACY'].values[0]
        drop = (baseline - sub_r['TEST_ACCURACY']) * 100  # percentage points
        ax.plot(sub_r['threshold'], drop,
                marker=markers[i % len(markers)], linewidth=2,
                label=f'γ={gamma}')

    ax.set_xlabel('Delta Threshold (th)')
    ax.set_ylabel('Accuracy Drop (pp)')
    ax.set_title('Accuracy Drop from LSTM Baseline')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, output_dir, 'pretrain_vs_retrain')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate DSE plots from merged log CSVs.')
    parser.add_argument('--dataset_name', default='gscdv2',
                        help='Dataset name (default: gscdv2)')
    parser.add_argument('--retrain_csv', default=None,
                        help='Path to merged retrain CSV (auto-discovered if omitted)')
    parser.add_argument('--pretrain_csv', default=None,
                        help='Path to merged pretrain CSV (auto-discovered if omitted)')
    parser.add_argument('--output_dir', default='./plot/dse',
                        help='Output directory for plots')
    parser.add_argument('--log_dir', default='./log',
                        help='Root log directory')
    parser.add_argument('--no_bit_sparsity', action='store_true',
                        help='Skip bit-level sparsity computation from saved models')
    parser.add_argument('--source', default='test', choices=['best', 'test'],
                        help='Which merged source to use (default: test)')
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load retrain data (primary) ---
    retrain_csv = args.retrain_csv or find_latest_merged(
        args.log_dir, args.dataset_name, 'retrain', args.source)
    if not retrain_csv:
        print(f"Error: no merged retrain CSV found. Run merge_logs.py first.")
        sys.exit(1)

    print(f"Loading retrain data: {retrain_csv}")
    df = load_and_prepare(retrain_csv)
    print(f"  {len(df)} configurations loaded")
    print(f"  Gammas:     {sorted(df['gamma'].unique())}")
    print(f"  Thresholds: {sorted(df['threshold'].unique())}")
    if df['msq_lambda'].nunique() > 1:
        print(f"  MSQ λ:      {sorted(df['msq_lambda'].unique())}")

    # --- Optionally add bit sparsity from saved models ---
    if 'bit_sp' in df.columns:
        print(f"  Bit sparsity loaded from CSV for {df['bit_sp'].notna().sum()}/{len(df)} configs")
    elif not args.no_bit_sparsity:
        print("Computing bit-level sparsity from saved models...")
        df = try_add_bit_sparsity(df, args.dataset_name)
        if 'bit_sp' in df.columns:
            print(f"  Bit sparsity computed for {df['bit_sp'].notna().sum()}/{len(df)} configs")
        else:
            print("  No saved models found; skipping bit-level analysis")

    # --- Compute speedup scenarios ---
    df = compute_speedup(df)

    # --- Load pretrain data (optional baseline) ---
    pretrain_csv = args.pretrain_csv or find_latest_merged(
        args.log_dir, args.dataset_name, 'pretrain', args.source)
    df_pretrain = None
    if pretrain_csv:
        print(f"Loading pretrain baseline: {pretrain_csv}")
        df_pretrain = load_and_prepare(pretrain_csv)
        print(f"  {len(df_pretrain)} pretrain configurations")

    # --- Generate all plots ---
    print(f"\nGenerating plots in {args.output_dir}/")

    # Category 1: Accuracy landscape
    plot_accuracy_heatmaps(df, args.output_dir)
    plot_accuracy_lines(df, args.output_dir)

    # Category 2: Sparsity decomposition
    plot_sparsity_heatmaps(df, args.output_dir)
    plot_sparsity_breakdown_bars(df, args.output_dir)
    plot_temporal_sparsity_lines(df, args.output_dir)

    # Category 3: Speedup analysis
    plot_speedup_heatmap(df, args.output_dir)
    plot_speedup_scenarios_bar(df, args.output_dir)

    # Category 4: Accuracy–speedup trade-off
    plot_accuracy_vs_speedup(df, args.output_dir)

    # Category 5: Bit-level sparsity (if available)
    plot_bit_sparsity(df, args.output_dir)
    plot_msq_comparison(df, args.output_dir)

    # Category 6: Summary dashboard
    plot_summary_dashboard(df, args.output_dir)

    # Category 7: Pretrain vs retrain comparison
    plot_pretrain_vs_retrain(df, df_pretrain, args.output_dir)

    # --- Summary ---
    n_plots = len(glob.glob(os.path.join(args.output_dir, '*.png')))
    print(f"\nDone! {n_plots} plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
