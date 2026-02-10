"""
Logging utilities for 3ET+ Event-Based Eye Tracking - Regression task.
Uses rich library for console output, adapted for regression metrics.
"""
import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Import model stats utility
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
    spatial_factor = getattr(args, 'spatial_factor', 0.125)
    n_time_bins = getattr(args, 'n_time_bins', 3)
    seq_length = getattr(args, 'seq_length', 30)

    dataset_id = '_D_' + args.dataset_name \
                 + '_SF_' + str(spatial_factor) \
                 + '_TB_' + str(n_time_bins) \
                 + '_SL_' + str(seq_length)

    trainfile = 'TRAIN' + dataset_id
    valfile = 'VAL' + dataset_id
    testfile = 'TEST' + dataset_id

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

    # Add CNN frontend flag
    if getattr(args, 'cnn_frontend', 0):
        dict_be['CNN'] = '1'

    # Add JaneEyeNet-specific tags
    if args.hid_type == 'JaneEyeNet':
        dict_be['JEN'] = '1'
        tbptt = getattr(args, 'tbptt', 0)
        if tbptt > 0:
            dict_be['TBPTT'] = f"{tbptt:d}"
        jen_drop = getattr(args, 'jen_drop_prob', 0.6)
        dict_be['JDROP'] = f"{jen_drop:.1f}"

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

    pixel_tolerances_str = getattr(args, 'pixel_tolerances', '5,10,15')
    pixel_tolerances = [int(x) for x in pixel_tolerances_str.split(',')]

    if val_stat is not None:
        list_log_headers.append('LOSS_VAL')
        list_log_values.append(val_stat['loss'])
        if args.score_val:
            list_log_headers.append('EUCLIDEAN_DIST_VAL')
            list_log_values.append(val_stat.get('euclidean_dist', 0))
            for p_tol in pixel_tolerances:
                list_log_headers.append(f'P{p_tol}_VAL')
                list_log_values.append(val_stat.get(f'p{p_tol}', 0))

    if test_stat is not None:
        list_log_headers.append('LOSS_TEST')
        list_log_values.append(test_stat['loss'])
        if args.score_test:
            list_log_headers.append('EUCLIDEAN_DIST_TEST')
            list_log_values.append(test_stat.get('euclidean_dist', 0))
            for p_tol in pixel_tolerances:
                list_log_headers.append(f'P{p_tol}_TEST')
                list_log_values.append(test_stat.get(f'p{p_tol}', 0))

    logger.add_row(list_log_headers, list_log_values)
    logger.write_csv()


def _get_color_for_dist(value):
    """Get color based on Euclidean distance (lower is better)."""
    if value <= 5.0:
        return "bright_green"
    elif value <= 10.0:
        return "green"
    elif value <= 15.0:
        return "yellow"
    elif value <= 20.0:
        return "orange1"
    else:
        return "red"


def _get_color_for_p_acc(value):
    """Get color based on p-accuracy (higher is better)."""
    if value >= 0.9:
        return "bright_green"
    elif value >= 0.7:
        return "green"
    elif value >= 0.5:
        return "yellow"
    elif value >= 0.3:
        return "orange1"
    else:
        return "red"


def print_log(args, log_stat, train_stat, val_stat, test_stat):
    """Print training log using rich formatting."""
    epoch = log_stat['EPOCH']
    n_epochs = log_stat['N_EPOCH']
    time_curr = log_stat['TIME_CURR']
    lr_curr = log_stat['LR_CURR']
    alpha = log_stat['ALPHA']
    sp_w_rnn = log_stat['SP_W_RNN']
    sp_w_fc = log_stat['SP_W_FC']

    console.print()

    # Epoch header
    header_text = Text()
    header_text.append(f"EPOCH {epoch + 1}/{n_epochs}", style="bold white")
    header_text.append(f"  |  Time: {time_curr}", style="dim")
    header_text.append(f"  |  LR: {lr_curr:.6f}", style="cyan")
    header_text.append(f"  |  Alpha: {alpha:.2f}", style="magenta")
    console.print(Panel(header_text, box=box.DOUBLE, border_style="blue"))

    # Determine which p-accuracy columns to show
    pixel_tolerances_str = getattr(args, 'pixel_tolerances', '5,10,15')
    pixel_tolerances = [int(x) for x in pixel_tolerances_str.split(',')]

    # Metrics table
    metrics_table = Table(title="Regression Metrics", box=box.ROUNDED,
                          show_header=True, header_style="bold cyan")
    metrics_table.add_column("Split", style="bold", width=8)
    metrics_table.add_column("Loss", justify="right", width=10)
    metrics_table.add_column("Euclid. Dist", justify="right", width=14)
    for p_tol in pixel_tolerances:
        metrics_table.add_column(f"p{p_tol} Acc", justify="right", width=10)

    # Train row
    if train_stat is not None:
        metrics_table.add_row(
            "Train",
            f"{train_stat.get('loss', 0):.6f}",
            "-",
            *(["-"] * len(pixel_tolerances))
        )

    # Validation row
    if val_stat is not None and args.score_val:
        ed = val_stat.get('euclidean_dist', 0)
        ed_color = _get_color_for_dist(ed)

        p_acc_vals = []
        for p_tol in pixel_tolerances:
            pv = val_stat.get(f'p{p_tol}', 0)
            pc = _get_color_for_p_acc(pv)
            p_acc_vals.append(f"[{pc}]{pv * 100:.1f}%[/]")

        metrics_table.add_row(
            "[bold green]Val[/]",
            f"{val_stat.get('loss', 0):.6f}",
            f"[{ed_color}]{ed:.2f} px[/]",
            *p_acc_vals
        )

    # Test row
    if test_stat is not None and args.score_test:
        ed = test_stat.get('euclidean_dist', 0)
        ed_color = _get_color_for_dist(ed)

        p_acc_vals = []
        for p_tol in pixel_tolerances:
            pv = test_stat.get(f'p{p_tol}', 0)
            pc = _get_color_for_p_acc(pv)
            p_acc_vals.append(f"[{pc}]{pv * 100:.1f}%[/]")

        metrics_table.add_row(
            "[bold blue]Test[/]",
            f"{test_stat.get('loss', 0):.6f}",
            f"[{ed_color}]{ed:.2f} px[/]",
            *p_acc_vals
        )

    console.print(metrics_table)

    # Sparsity info
    sparsity_text = Text()
    sparsity_text.append("Sparsity: ", style="bold")
    sparsity_text.append(f"RNN Weights: {sp_w_rnn * 100:.2f}%", style="yellow")
    sparsity_text.append("  |  ", style="dim")
    sparsity_text.append(f"FC Weights: {sp_w_fc * 100:.2f}%", style="yellow")
    console.print(sparsity_text)


def print_training_header(args, n_features, n_classes, n_param):
    """Print training header with model info."""
    console.print()
    header_table = Table(title="Model Configuration", box=box.DOUBLE_EDGE, show_header=False)
    header_table.add_column("Property", style="bold cyan")
    header_table.add_column("Value", style="white")

    header_table.add_row("Dataset", "3ET+ Event-Based Eye Tracking")
    header_table.add_row("Task", "Pupil Center Regression (x, y)")
    header_table.add_row("Model Type", args.hid_type)
    header_table.add_row("CNN Frontend", "Yes" if getattr(args, 'cnn_frontend', 0) else "No")
    header_table.add_row("Hidden Layers", str(args.hid_layers))
    header_table.add_row("Hidden Size", str(args.hid_size))
    header_table.add_row("Input Features", str(n_features))
    header_table.add_row("Output Dims", str(n_classes))
    header_table.add_row("Total Parameters", f"{n_param:,}")
    header_table.add_row("Batch Size", str(args.batch_size))
    header_table.add_row("Learning Rate", str(args.lr))
    header_table.add_row("Epochs", str(args.n_epochs))

    console.print(header_table)
    console.print()


def print_model_analysis(net, args, n_features, seq_len=30, device='cpu'):
    """Print detailed model analysis."""
    if not MODEL_STATS_AVAILABLE:
        console.print("[yellow]Model stats not available. Skipping detailed analysis.[/]")
        return

    console.print()

    try:
        model_name = f"{args.hid_type} ({args.hid_layers}L x {args.hid_size}H)"
        if getattr(args, 'cnn_frontend', 0):
            model_name = f"CNN + {model_name}"

        stats = analyze_rnn_model(
            net,
            seq_len=seq_len,
            n_features=n_features,
            batch_size=1,
            device=device,
            model_name=model_name,
            args=args,
            print_stats=True
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

        delta_table.add_row("Input Delta Sparsity (dx)", f"{sp_dx*100:.2f}%")
        delta_table.add_row("Hidden Delta Sparsity (dh)", f"{sp_dh*100:.2f}%")
        delta_table.add_row("Combined Sparsity (dxh)", f"[bold]{sp_dxh*100:.2f}%[/]")

        console.print(delta_table)

        bar_width = 40
        filled_x = int(sp_dx * bar_width)
        filled_h = int(sp_dh * bar_width)
        filled_xh = int(sp_dxh * bar_width)

        console.print()
        console.print(f"  dx:  [green]{'█' * filled_x}[/][dim]{'░' * (bar_width - filled_x)}[/] {sp_dx*100:.1f}%")
        console.print(f"  dh:  [green]{'█' * filled_h}[/][dim]{'░' * (bar_width - filled_h)}[/] {sp_dh*100:.1f}%")
        console.print(f"  dxh: [cyan]{'█' * filled_xh}[/][dim]{'░' * (bar_width - filled_xh)}[/] {sp_dxh*100:.1f}%")

    except Exception as e:
        console.print(f"[red]Failed to get delta stats: {e}[/]")


def print_training_complete(best_val_dist, best_epoch):
    """Print training completion message."""
    console.print()
    complete_panel = Panel(
        Text.assemble(
            ("Training Complete!\n\n", "bold green"),
            ("Best Val Euclidean Dist: ", "bold"),
            (f"{best_val_dist:.2f} px", "bold cyan"),
            ("\nBest Epoch: ", "bold"),
            (f"{best_epoch + 1}", "bold cyan")
        ),
        title="[bold green]Done[/]",
        border_style="green",
        box=box.DOUBLE
    )
    console.print(complete_panel)


def save_best_model(args, best_val, net, save_file, logger, logfile_best, epoch, val_stat, score_val):
    """Save best model based on validation metric.

    Supports best_model_criterion:
      - 'euclidean_dist': lower is better (default)
      - 'p3', 'p5', 'p10', 'p15': higher is better
    """
    best_criterion = getattr(args, 'best_model_criterion', 'euclidean_dist')

    if score_val:
        if best_criterion in ('p3', 'p5', 'p10', 'p15'):
            # p-accuracy: higher is better
            current_metric = val_stat.get(best_criterion, 0.0)
            is_better = (epoch == 0 or best_val is None or current_metric >= best_val)
        else:
            # Euclidean distance: lower is better
            current_metric = val_stat.get('euclidean_dist', val_stat.get('per', float('inf')))
            is_better = (epoch == 0 or best_val is None or current_metric <= best_val)

        if is_better:
            best_val = current_metric
            torch.save(net.state_dict(), save_file)
            logger.write_log_idx(epoch, logfile_best)
            console.print(f'[bold green]>>> Saving best model from epoch {epoch} to {save_file}[/]')
    else:
        if epoch == 0 or best_val is None:
            best_val = val_stat['loss']
            torch.save(net.state_dict(), save_file)
            logger.write_log_idx(epoch, logfile_best)
            console.print(f'[bold green]>>> Saving best model from epoch {epoch} to {save_file}[/]')
        elif val_stat['loss'] <= best_val:
            best_val = val_stat['loss']
            logger.write_log_idx(epoch, logfile_best)
            torch.save(net.state_dict(), save_file)
            console.print(f'[bold green]>>> Saving best model from epoch {epoch} to {save_file}[/]')

    return best_val
