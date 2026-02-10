"""
Logging utilities for Google Speech Commands Dataset v2 - Keyword Spotting.
Uses rich library for beautiful console output with enhanced model analysis.
Integrates torchstat2 for comprehensive model statistics.
"""
import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

# Import model stats utility (integrates torchstat2)
try:
    from utils.model_stats import (
        analyze_rnn_model, get_model_summary_dict,
        run_throughput_analysis, format_number,
        RNNModelStats, print_model_stats_rich
    )
    MODEL_STATS_AVAILABLE = True
except ImportError:
    MODEL_STATS_AVAILABLE = False

# Initialize console
console = Console()

# 12-class KWS labels
KWS_LABELS = ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.list_header = []
        self.loglist = []

    def add_row(self, list_header, list_value):
        self.list_header = list_header
        row = {}
        for header, value in zip(list_header, list_value):
            row[header] = value
        self.loglist.append(row)

    def write_log(self, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined.", RuntimeWarning)
        else:
            df = pd.DataFrame(self.loglist, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)

    def write_csv(self):
        """Alias for write_log."""
        self.write_log()

    def write_log_idx(self, idx, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined.", RuntimeWarning)
        else:
            loglist_best = [self.loglist[idx]]
            df = pd.DataFrame(loglist_best, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)


def gen_dataset_name(args):
    """Generate dataset file names."""
    dataset_id = '_D_' + args.dataset_name \
                 + '_NF_' + str(args.n_filt) \
                 + '_SI_' + str(args.frame_size) \
                 + '_ST_' + str(args.frame_stride) \
                 + '_GC_' + str(args.gain_ctrl) \
                 + '_FG_' + str(args.gradient)

    trainfile = 'TRAIN' + dataset_id + '.h5'
    valfile = 'VAL' + dataset_id + '.h5'
    testfile = 'TEST' + dataset_id + '.h5'

    return dataset_id, trainfile, valfile, testfile


def gen_model_id(args, n_features):
    """Generate model ID for saving."""
    dict_be = {
        'SD': f"{args.seed:d}",
        'L': f"{args.hid_layers:d}",
        'IN': f"{n_features:d}",
        'H': f"{args.hid_size:d}",
        'CLA': args.hid_type,
        'FC': str(args.fc_extra_size),
        'QA': f"{args.qa:d}",
        'AI': f"{args.aqi:d}",
        'AF': f"{args.aqf:d}",
        'QW': f"{args.qw:d}",
        'WI': f"{args.wqi:d}",
        'WF': f"{args.wqf:d}",
    }
    dict_be_pretrain = dict(dict_be.items())
    dict_be_pretrain['CLA'] = args.hid_type_pretrain

    dict_cbwdrop0 = {
        'CBWD': f"{args.cbtd:d}",
        'SPRNN': f"{args.gamma_rnn:.2f}",
        'SPFC': f"{args.gamma_fc:.2f}",
        'AAE': f"{args.alpha_anneal_epoch:d}",
    }

    if args.balance_pe:
        dict_cbwdrop1 = {'NAP': f"{args.num_array_pe:d}"}
    else:
        dict_cbwdrop1 = {'NAP': f"{0:d}"}

    dict_deltanet = {
        'TX': f"{args.th_x:.2f}",
        'TH': f"{args.th_h:.2f}",
        'BT': f"{args.beta:.1e}",
    }

    dict_model_id = dict(list(dict_be.items()) + list(dict_cbwdrop0.items()) + list(dict_cbwdrop1.items()))

    if args.phase == 'pretrain':
        dict_pretrain_model_id = None
    elif args.phase == 'retrain':
        dict_pretrain_model_id = dict(
            list(dict_be_pretrain.items()) + list(dict_cbwdrop0.items()) + list(dict_cbwdrop1.items()))
    else:
        dict_pretrain_model_id = None

    if 'Delta' in args.hid_type or args.hid_type in ['dLSTM']:
        dict_model_id = dict(list(dict_model_id.items()) + list(dict_deltanet.items()))

    if getattr(args, 'msq_lambda', 0) > 0:
        dict_model_id['MSQL'] = f"{args.msq_lambda:.0e}"
        dict_model_id['MSQK'] = f"{getattr(args, 'msq_k', 2):d}"

    list_model_id = []
    for item in list(dict_model_id.items()):
        list_model_id += list(item)
    model_id = '_'.join(list_model_id)

    if dict_pretrain_model_id is not None:
        list_pretrain_model_id = []
        for item in list(dict_pretrain_model_id.items()):
            list_pretrain_model_id += list(item)
        pretrain_model_id = '_'.join(list_pretrain_model_id)
    else:
        pretrain_model_id = None

    return model_id, pretrain_model_id


def gen_test_id(args, n_features, n_classes):
    """Generate test ID."""
    return gen_model_id(args, n_features)


def write_log(args, logger, tb_writer, model_id, train_stat, val_stat, test_stat, net, optimizer, epoch,
              time_curr, alpha, retrain):
    """Write training log."""
    n_param = sum(p.numel() for p in net.parameters())

    # Evaluate RNN weight sparsity
    n_nonzero_weight_elem = 0
    n_weight_elem = 0
    for name, param in net.named_parameters():
        if 'rnn' in name and 'weight' in name:
            n_nonzero_weight_elem += len(param.data.nonzero())
            n_weight_elem += param.data.nelement()
    sp_w_rnn = 1 - (n_nonzero_weight_elem / n_weight_elem) if n_weight_elem > 0 else 0

    # Evaluate FC weight sparsity
    sp_w_fc = 0
    if args.fc_extra_size:
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in net.named_parameters():
            if 'fc_extra' in name and 'weight' in name:
                n_nonzero_weight_elem += len(param.data.nonzero())
                n_weight_elem += param.data.nelement()
        sp_w_fc = 1 - (n_nonzero_weight_elem / n_weight_elem) if n_weight_elem > 0 else 0

    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    list_log_headers = ['EPOCH', 'TIME', 'SEED', 'N_PARAM', 'BATCH_SIZE',
                        'LR', 'SP_W_RNN', 'SP_W_FC', 'ALPHA']
    list_log_values = [epoch + 1, time_curr, args.seed, n_param, args.batch_size,
                       lr_curr, sp_w_rnn, sp_w_fc, alpha]

    if train_stat is not None:
        list_log_headers.extend(['LOSS_TRAIN', 'REG_TRAIN'])
        list_log_values.extend([train_stat['loss'], train_stat['reg']])

    if val_stat is not None:
        list_log_headers.append('LOSS_VAL')
        list_log_values.append(val_stat['loss'])
        if args.score_val:
            list_log_headers.extend(['ACC_VAL', 'ERR_VAL', 'F1_VAL', 'FAR_VAL', 'EER_VAL', 'AUC_VAL'])
            acc = val_stat.get('accuracy', 1 - val_stat.get('per', 0))
            err = val_stat.get('error_rate', val_stat.get('per', 0))
            f1 = val_stat.get('f1_macro', 0)
            far = val_stat.get('far_macro', 0)
            eer = val_stat.get('eer_macro', 0)
            auc_val = val_stat.get('auc_macro', 0)
            list_log_values.extend([acc, err, f1, far, eer, auc_val])

    if test_stat is not None:
        list_log_headers.append('LOSS_TEST')
        list_log_values.append(test_stat['loss'])
        if args.score_test:
            list_log_headers.extend(['ACC_TEST', 'ERR_TEST', 'F1_TEST', 'FAR_TEST', 'EER_TEST', 'AUC_TEST'])
            acc = test_stat.get('accuracy', 1 - test_stat.get('per', 0))
            err = test_stat.get('error_rate', test_stat.get('per', 0))
            f1 = test_stat.get('f1_macro', 0)
            far = test_stat.get('far_macro', 0)
            eer = test_stat.get('eer_macro', 0)
            auc_test = test_stat.get('auc_macro', 0)
            list_log_values.extend([acc, err, f1, far, eer, auc_test])

    logger.add_row(list_log_headers, list_log_values)
    logger.write_csv()


def _create_metric_bar(value, max_value=1.0, width=20, color="green"):
    """Create a text-based progress bar for metrics."""
    filled = int((value / max_value) * width)
    bar = "[" + color + "]" + "█" * filled + "[/]" + "░" * (width - filled)
    return bar


def _get_color_for_metric(value, metric_type="accuracy"):
    """Get color based on metric value."""
    if metric_type == "accuracy" or metric_type == "auc":
        if value >= 0.95:
            return "bright_green"
        elif value >= 0.90:
            return "green"
        elif value >= 0.80:
            return "yellow"
        elif value >= 0.70:
            return "orange1"
        else:
            return "red"
    elif metric_type == "error" or metric_type == "far" or metric_type == "eer":
        if value <= 0.05:
            return "bright_green"
        elif value <= 0.10:
            return "green"
        elif value <= 0.20:
            return "yellow"
        elif value <= 0.30:
            return "orange1"
        else:
            return "red"
    return "white"


def print_rich_epoch_summary(epoch, n_epochs, time_curr, lr_curr, alpha, sp_w_rnn, sp_w_fc,
                             train_stat, val_stat, test_stat, args):
    """Print a rich formatted epoch summary with all metrics."""
    console.print()

    # Epoch header panel
    header_text = Text()
    header_text.append(f"EPOCH {epoch + 1}/{n_epochs}", style="bold white")
    header_text.append(f"  |  Time: {time_curr}", style="dim")
    header_text.append(f"  |  LR: {lr_curr:.6f}", style="cyan")
    header_text.append(f"  |  Alpha: {alpha:.2f}", style="magenta")
    console.print(Panel(header_text, box=box.DOUBLE, border_style="blue"))

    # Create main metrics table
    metrics_table = Table(title="Training Metrics", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    metrics_table.add_column("Split", style="bold", width=8)
    metrics_table.add_column("Loss", justify="right", width=10)
    metrics_table.add_column("Accuracy", justify="right", width=12)
    metrics_table.add_column("F1 (Macro)", justify="right", width=12)
    metrics_table.add_column("FAR", justify="right", width=10)
    metrics_table.add_column("EER", justify="right", width=10)
    metrics_table.add_column("AUC-ROC", justify="right", width=12)

    # Train row
    if train_stat is not None:
        metrics_table.add_row(
            "Train",
            f"{train_stat.get('loss', 0):.4f}",
            "-", "-", "-", "-", "-"
        )

    # Validation row
    if val_stat is not None and args.score_val:
        acc = val_stat.get('accuracy', 0)
        f1 = val_stat.get('f1_macro', 0)
        far = val_stat.get('far_macro', 0)
        eer = val_stat.get('eer_macro', 0)
        auc_val = val_stat.get('auc_macro', 0)

        acc_color = _get_color_for_metric(acc, "accuracy")
        auc_color = _get_color_for_metric(auc_val, "auc")
        eer_color = _get_color_for_metric(eer, "eer")
        far_color = _get_color_for_metric(far, "far")

        metrics_table.add_row(
            "[bold green]Val[/]",
            f"{val_stat.get('loss', 0):.4f}",
            f"[{acc_color}]{acc * 100:.2f}%[/]",
            f"{f1:.4f}",
            f"[{far_color}]{far * 100:.2f}%[/]",
            f"[{eer_color}]{eer * 100:.2f}%[/]",
            f"[{auc_color}]{auc_val:.4f}[/]"
        )

    # Test row
    if test_stat is not None and args.score_test:
        acc = test_stat.get('accuracy', 0)
        f1 = test_stat.get('f1_macro', 0)
        far = test_stat.get('far_macro', 0)
        eer = test_stat.get('eer_macro', 0)
        auc_test = test_stat.get('auc_macro', 0)

        acc_color = _get_color_for_metric(acc, "accuracy")
        auc_color = _get_color_for_metric(auc_test, "auc")
        eer_color = _get_color_for_metric(eer, "eer")
        far_color = _get_color_for_metric(far, "far")

        metrics_table.add_row(
            "[bold blue]Test[/]",
            f"{test_stat.get('loss', 0):.4f}",
            f"[{acc_color}]{acc * 100:.2f}%[/]",
            f"{f1:.4f}",
            f"[{far_color}]{far * 100:.2f}%[/]",
            f"[{eer_color}]{eer * 100:.2f}%[/]",
            f"[{auc_color}]{auc_test:.4f}[/]"
        )

    console.print(metrics_table)

    # Sparsity info
    sparsity_text = Text()
    sparsity_text.append("Sparsity: ", style="bold")
    sparsity_text.append(f"RNN Weights: {sp_w_rnn * 100:.2f}%", style="yellow")
    sparsity_text.append("  |  ", style="dim")
    sparsity_text.append(f"FC Weights: {sp_w_fc * 100:.2f}%", style="yellow")
    console.print(sparsity_text)


def print_rich_per_class_metrics(val_stat, test_stat, labels=None):
    """Print per-class metrics in a rich table."""
    if labels is None:
        labels = KWS_LABELS

    # Per-class table
    class_table = Table(title="Per-Class Metrics", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    class_table.add_column("Class", style="bold", width=12)
    class_table.add_column("Val Acc", justify="right", width=10)
    class_table.add_column("Val FAR", justify="right", width=10)
    class_table.add_column("Val EER", justify="right", width=10)
    class_table.add_column("Val AUC", justify="right", width=10)
    class_table.add_column("Test Acc", justify="right", width=10)
    class_table.add_column("Test FAR", justify="right", width=10)
    class_table.add_column("Test EER", justify="right", width=10)
    class_table.add_column("Test AUC", justify="right", width=10)

    for label in labels:
        val_acc = val_stat.get('per_class_acc', {}).get(label, 0) if val_stat else 0
        val_far = val_stat.get('far_per_class', {}).get(label, 0) if val_stat else 0
        val_eer = val_stat.get('eer_per_class', {}).get(label, 0) if val_stat else 0
        val_auc = val_stat.get('auc_per_class', {}).get(label, 0) if val_stat else 0

        test_acc = test_stat.get('per_class_acc', {}).get(label, 0) if test_stat else 0
        test_far = test_stat.get('far_per_class', {}).get(label, 0) if test_stat else 0
        test_eer = test_stat.get('eer_per_class', {}).get(label, 0) if test_stat else 0
        test_auc = test_stat.get('auc_per_class', {}).get(label, 0) if test_stat else 0

        class_table.add_row(
            label,
            f"{val_acc * 100:.1f}%",
            f"{val_far * 100:.2f}%",
            f"{val_eer * 100:.2f}%",
            f"{val_auc:.3f}",
            f"{test_acc * 100:.1f}%",
            f"{test_far * 100:.2f}%",
            f"{test_eer * 100:.2f}%",
            f"{test_auc:.3f}"
        )

    console.print(class_table)


def print_rich_auc_bar_chart(stat, split_name="Val", labels=None):
    """Print AUC-ROC as a horizontal bar chart using rich."""
    if labels is None:
        labels = KWS_LABELS

    auc_per_class = stat.get('auc_per_class', {})
    if not auc_per_class:
        return

    console.print()
    console.print(f"[bold cyan]{split_name} AUC-ROC per Class[/]")
    console.print("=" * 60)

    for label in labels:
        auc_val = auc_per_class.get(label, 0)
        bar_width = 40
        filled = int(auc_val * bar_width)
        color = _get_color_for_metric(auc_val, "auc")

        bar = "█" * filled + "░" * (bar_width - filled)
        console.print(f"  {label:>10} [{color}]{bar}[/] {auc_val:.4f}")

    macro_auc = stat.get('auc_macro', 0)
    console.print("-" * 60)
    macro_color = _get_color_for_metric(macro_auc, "auc")
    console.print(f"  {'Macro Avg':>10} [bold {macro_color}]{macro_auc:.4f}[/]")


def print_rich_confusion_matrix(cm, labels=None):
    """Print confusion matrix using rich."""
    if labels is None:
        labels = KWS_LABELS

    cm_table = Table(title="Confusion Matrix", box=box.SQUARE, show_header=True,
                     header_style="bold white on dark_blue")

    # Add header column
    cm_table.add_column("True\\Pred", style="bold", width=10)
    for label in labels:
        cm_table.add_column(label[:6], justify="center", width=7)

    # Add rows
    for i, label in enumerate(labels):
        row_values = [label[:10]]
        row_max = cm[i].max() if cm[i].max() > 0 else 1
        for j in range(len(labels)):
            val = cm[i, j]
            if i == j:
                # Diagonal (correct predictions)
                intensity = val / row_max if row_max > 0 else 0
                if intensity > 0.7:
                    style = "bold green"
                elif intensity > 0.3:
                    style = "green"
                else:
                    style = "dim green"
                row_values.append(f"[{style}]{val}[/]")
            elif val > 0:
                # Off-diagonal (errors)
                row_values.append(f"[red]{val}[/]")
            else:
                row_values.append(f"[dim]{val}[/]")
        cm_table.add_row(*row_values)

    console.print(cm_table)


def print_log(args, log_stat, train_stat, val_stat, test_stat):
    """Print training log using rich formatting."""
    epoch = log_stat['EPOCH']
    n_epochs = log_stat['N_EPOCH']
    time_curr = log_stat['TIME_CURR']
    lr_curr = log_stat['LR_CURR']
    alpha = log_stat['ALPHA']
    sp_w_rnn = log_stat['SP_W_RNN']
    sp_w_fc = log_stat['SP_W_FC']

    # Print rich epoch summary
    print_rich_epoch_summary(
        epoch, n_epochs, time_curr, lr_curr, alpha, sp_w_rnn, sp_w_fc,
        train_stat, val_stat, test_stat, args
    )

    # Print per-class metrics at the end of each epoch
    if (val_stat and args.score_val) or (test_stat and args.score_test):
        print_rich_per_class_metrics(val_stat, test_stat)

    # Print AUC bar charts
    if val_stat and args.score_val and val_stat.get('auc_per_class'):
        print_rich_auc_bar_chart(val_stat, "Validation")

    if test_stat and args.score_test and test_stat.get('auc_per_class'):
        print_rich_auc_bar_chart(test_stat, "Test")


def print_training_header(args, n_features, n_classes, n_param):
    """Print training header with model info."""
    console.print()
    header_table = Table(title="Model Configuration", box=box.DOUBLE_EDGE, show_header=False)
    header_table.add_column("Property", style="bold cyan")
    header_table.add_column("Value", style="white")

    header_table.add_row("Dataset", args.dataset_name)
    header_table.add_row("Model Type", args.hid_type)
    header_table.add_row("Hidden Layers", str(args.hid_layers))
    header_table.add_row("Hidden Size", str(args.hid_size))
    header_table.add_row("Input Features", str(n_features))
    header_table.add_row("Output Classes", str(n_classes))
    header_table.add_row("Total Parameters", f"{n_param:,}")
    header_table.add_row("Batch Size", str(args.batch_size))
    header_table.add_row("Learning Rate", str(args.lr))
    header_table.add_row("Epochs", str(args.n_epochs))

    console.print(header_table)
    console.print()


def print_model_analysis(net, args, n_features, seq_len=98, device='cpu'):
    """
    Print detailed model analysis using torchstat2-based model stats.

    Args:
        net: PyTorch model
        args: Arguments namespace
        n_features: Number of input features
        seq_len: Sequence length for RNN models
        device: Device to run analysis on
    """
    if not MODEL_STATS_AVAILABLE:
        console.print("[yellow]Model stats not available. Skipping detailed analysis.[/]")
        return

    console.print()

    try:
        # Use the new model_stats module
        model_name = f"{args.hid_type} ({args.hid_layers}L x {args.hid_size}H)"
        stats = analyze_rnn_model(
            net,
            seq_len=seq_len,
            n_features=n_features,
            batch_size=1,
            device=device,
            model_name=model_name,
            args=args,
            print_stats=True  # Will use rich formatting
        )

        return stats

    except Exception as e:
        console.print(f"[red]Model analysis failed: {e}[/]")
        import traceback
        traceback.print_exc()
        return None


def print_sparsity_analysis(net, args):
    """Print detailed sparsity analysis for the model."""
    console.print()
    console.print(Panel("[bold yellow]Weight Sparsity Analysis[/]", border_style="yellow"))

    sparsity_table = Table(box=box.ROUNDED, show_header=True, header_style="bold yellow")
    sparsity_table.add_column("Layer", style="bold", width=35)
    sparsity_table.add_column("Shape", justify="center", width=20)
    sparsity_table.add_column("Total", justify="right", width=12)
    sparsity_table.add_column("Non-zero", justify="right", width=12)
    sparsity_table.add_column("Sparsity", justify="right", width=10)

    total_params = 0
    total_nonzero = 0

    for name, param in net.named_parameters():
        if 'weight' in name:
            n_total = param.data.numel()
            n_nonzero = torch.count_nonzero(param.data).item()
            sparsity = 1 - (n_nonzero / n_total) if n_total > 0 else 0

            total_params += n_total
            total_nonzero += n_nonzero

            # Color code sparsity
            if sparsity > 0.8:
                sp_style = "green"
            elif sparsity > 0.5:
                sp_style = "yellow"
            elif sparsity > 0.2:
                sp_style = "orange1"
            else:
                sp_style = "white"

            sparsity_table.add_row(
                name[:35],
                str(list(param.shape)),
                f"{n_total:,}",
                f"{n_nonzero:,}",
                f"[{sp_style}]{sparsity*100:.1f}%[/]"
            )

    # Add totals row
    overall_sparsity = 1 - (total_nonzero / total_params) if total_params > 0 else 0
    sparsity_table.add_row("", "", "", "", "")
    sparsity_table.add_row(
        "[bold]TOTAL[/]",
        "",
        f"[bold]{total_params:,}[/]",
        f"[bold]{total_nonzero:,}[/]",
        f"[bold]{overall_sparsity*100:.1f}%[/]"
    )

    console.print(sparsity_table)


def print_delta_stats(net, test_stat=None):
    """Print DeltaLSTM temporal sparsity statistics."""
    if not hasattr(net, 'rnn') or not hasattr(net.rnn, 'get_temporal_sparsity'):
        return

    console.print()
    console.print(Panel("[bold orange1]Temporal Sparsity Statistics[/]", border_style="orange1"))

    try:
        # Use test_stat if available (from actual evaluation), otherwise use model's internal stats
        if test_stat and 'sp_dx' in test_stat:
            sp_dx = test_stat['sp_dx']
            sp_dh = test_stat['sp_dh']
            sp_dxh = test_stat.get('sp_dxh', 0)
        else:
            stats = net.rnn.get_temporal_sparsity()
            sp_dx = stats.get('sparsity_delta_x', 0)
            sp_dh = stats.get('sparsity_delta_h', 0)
            sp_dxh = stats.get('sparsity_delta', 0)

        delta_table = Table(box=box.ROUNDED, show_header=False)
        delta_table.add_column("Metric", style="bold orange1", width=30)
        delta_table.add_column("Value", style="white", width=15)

        delta_table.add_row("Input Delta Sparsity (Δx)", f"{sp_dx*100:.2f}%")
        delta_table.add_row("Hidden Delta Sparsity (Δh)", f"{sp_dh*100:.2f}%")
        delta_table.add_row("Combined Sparsity (Δxh)", f"[bold]{sp_dxh*100:.2f}%[/]")

        console.print(delta_table)

        # Visualize sparsity as bar
        bar_width = 40
        filled_x = int(sp_dx * bar_width)
        filled_h = int(sp_dh * bar_width)
        filled_xh = int(sp_dxh * bar_width)

        console.print()
        console.print(f"  Δx:  [green]{'█' * filled_x}[/][dim]{'░' * (bar_width - filled_x)}[/] {sp_dx*100:.1f}%")
        console.print(f"  Δh:  [green]{'█' * filled_h}[/][dim]{'░' * (bar_width - filled_h)}[/] {sp_dh*100:.1f}%")
        console.print(f"  Δxh: [cyan]{'█' * filled_xh}[/][dim]{'░' * (bar_width - filled_xh)}[/] {sp_dxh*100:.1f}%")

    except Exception as e:
        console.print(f"[red]Failed to get delta stats: {e}[/]")


def print_training_complete(best_val_acc, best_epoch):
    """Print training completion message."""
    console.print()
    complete_panel = Panel(
        Text.assemble(
            ("Training Complete!\n\n", "bold green"),
            ("Best Validation Accuracy: ", "bold"),
            (f"{best_val_acc * 100:.2f}%", "bold cyan"),
            ("\nBest Epoch: ", "bold"),
            (f"{best_epoch + 1}", "bold cyan")
        ),
        title="[bold green]Done[/]",
        border_style="green",
        box=box.DOUBLE
    )
    console.print(complete_panel)


def save_best_model(args, best_val, net, save_file, logger, logfile_best, epoch, val_stat, score_val):
    """Save best model based on validation accuracy."""
    if score_val:
        # Use error rate (lower is better)
        current_metric = val_stat.get('error_rate', val_stat.get('per', 1.0))
        if epoch == 0:
            best_val = current_metric
            torch.save(net.state_dict(), save_file)
            logger.write_log_idx(epoch, logfile_best)
            console.print(f'[bold green]>>> Saving best model from epoch {epoch} to {save_file}[/]')
        if current_metric <= best_val:
            best_val = current_metric
            logger.write_log_idx(epoch, logfile_best)
            torch.save(net.state_dict(), save_file)
            console.print(f'[bold green]>>> Saving best model from epoch {epoch} to {save_file}[/]')
    else:
        if epoch == 0:
            best_val = val_stat['loss']
            torch.save(net.state_dict(), save_file)
            logger.write_log_idx(epoch, logfile_best)
            console.print(f'[bold green]>>> Saving best model from epoch {epoch} to {save_file}[/]')
        if val_stat['loss'] <= best_val:
            best_val = val_stat['loss']
            logger.write_log_idx(epoch, logfile_best)
            torch.save(net.state_dict(), save_file)
            console.print(f'[bold green]>>> Saving best model from epoch {epoch} to {save_file}[/]')

    return best_val
