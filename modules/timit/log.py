"""
Logging utilities for TIMIT phoneme recognition.
Integrates torchstat2 for comprehensive model statistics.
"""
import warnings
import torch
import torch.nn as nn
import pandas as pd

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

# Import model stats utility (integrates torchstat2)
try:
    from utils.model_stats import (
        analyze_rnn_model, get_model_summary_dict,
        format_number, RNNModelStats
    )
    MODEL_STATS_AVAILABLE = True
except ImportError:
    MODEL_STATS_AVAILABLE = False


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
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            df = pd.DataFrame(self.loglist, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)

    def write_log_idx(self, idx, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            loglist_best = [self.loglist[idx]]
            df = pd.DataFrame(loglist_best, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)


def gen_dataset_name(args):
    # Feat name
    dataset_id = '_D_' + args.dataset_name \
                 + '_NF_' + str(args.n_filt) \
                 + '_SI_' + str(args.frame_size) \
                 + '_ST_' + str(args.frame_stride) \
                 + '_GC_' + str(args.gain_ctrl) \
                 + '_FG_' + str(args.gradient)

    trainfile = 'TRAIN' + dataset_id
    valfile = 'VAL' + dataset_id
    testfile = 'TEST' + dataset_id

    trainfile += '.h5'
    valfile += '.h5'
    testfile += '.h5'

    return dataset_id, trainfile, valfile, testfile


def gen_model_id(args, n_features):
    # Collect Basic Model ID Elements
    dict_be = {'SD': f"{args.seed:d}",
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

    # Collect Column-Balanced Pruning Sparsity Elements
    dict_cbwdrop0 = {'CBWD': f"{args.cbtd:d}",
                     'SPRNN': f"{args.gamma_rnn:.2f}",
                     'SPFC': f"{args.gamma_fc:.2f}",
                     'AAE': f"{args.alpha_anneal_epoch:d}",
                     }
    if args.balance_pe:
        dict_cbwdrop1 = {'NAP': f"{args.num_array_pe:d}"
                         }
    else:
        dict_cbwdrop1 = {'NAP': f"{0:d}"}

    # Delta Network
    dict_deltanet = {'TX': f"{args.th_x:.2f}",
                     'TH': f"{args.th_h:.2f}",
                     'BT': f"{args.beta:.1e}",
                     }

    # Pretrain Model ID
    dict_model_id = dict(list(dict_be.items()) + list(dict_cbwdrop0.items()) + list(dict_cbwdrop1.items()))
    if args.phase == 'pretrain':
        dict_pretrain_model_id = None
    elif args.phase == 'retrain':
        dict_pretrain_model_id = dict(
            list(dict_be_pretrain.items()) + list(dict_cbwdrop0.items()) + list(dict_cbwdrop1.items()))
    else:
        dict_pretrain_model_id = None

    # Add info for Delta Networks
    if 'Delta' in args.hid_type or args.hid_type in ['dLSTM']:
        dict_model_id = dict(list(dict_model_id.items()) + list(dict_deltanet.items()))

    if getattr(args, 'msq_lambda', 0) > 0:
        dict_model_id['MSQL'] = f"{args.msq_lambda:.0e}"
        dict_model_id['MSQK'] = f"{getattr(args, 'msq_k', 2):d}"

    # Model ID
    list_model_id = []
    for item in list(dict_model_id.items()):
        list_model_id += list(item)
    model_id = '_'.join(list_model_id)

    # Pretrain Model ID
    if dict_pretrain_model_id is not None:
        list_pretrain_model_id = []
        for item in list(dict_pretrain_model_id.items()):
            list_pretrain_model_id += list(item)
        pretrain_model_id = '_'.join(list_pretrain_model_id)
    else:
        pretrain_model_id = None

    return model_id, pretrain_model_id


def gen_test_id(args, n_features, n_classes):
    # Custom String
    str_custom = '' if args.filename == '' else args.filename + '_'

    # Collect Basic Model ID Elements
    dict_be = {'SD': f"{args.seed:d}",
               # 'L': f"{args.hid_layers:d}",
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

    # Collect Column-Balanced Pruning Sparsity Elements
    dict_cbwdrop0 = {'CBWD': f"{args.cbtd:d}",
                     'SPRNN': f"{args.gamma_rnn:.2f}",
                     'SPFC': f"{args.gamma_fc:.2f}",
                     'AAE': f"{args.alpha_anneal_epoch:d}",
                     }
    if args.balance_pe:
        dict_cbwdrop1 = {'NAP': f"{args.num_array_pe:d}",
                         'NA': f"{args.num_array:d}"
                         }
    else:
        dict_cbwdrop1 = {'NAP': f"{0:d}"}

    # Delta Network
    dict_deltanet = {'TX': f"{args.th_x:.2f}",
                     'TH': f"{args.th_h:.2f}",
                     'BT': f"{args.beta:.1e}",
                     }
    #
    # Quantization of Gradients
    # if args.hid_type in ['qLSTM', 'dLSTM']:
    #     str_net_arch += '_QG_' + str(args.qg) \
    #                     + '_GQI_' + f"{args.gqi:d}" \
    #                     + '_GQF_' + f"{args.gqf:d}"
    # if args.fc_type in ['qFC']:
    #     if args.fc_extra_size:
    #         str_net_arch += '_QA1_' + str(args.qa_fc_extra) \
    #                         + '_A1QI_' + f"{args.aqi_fc_extra:d}" \
    #                         + '_A1QF_' + f"{args.aqf_fc_extra:d}"
    #     str_net_arch += '_QA2_' + str(args.qa_fc_final) \
    #                     + '_A2QI_' + f"{args.aqi_fc_final:d}" \
    #                     + '_A2QF_' + f"{args.aqf_fc_final:d}"
    # Column Balanced Weight Dropout

    # Pretrain Model ID
    dict_model_id = dict(list(dict_be.items()) + list(dict_cbwdrop0.items()) + list(dict_cbwdrop1.items()))
    if args.phase == 'pretrain':
        dict_pretrain_model_id = None
    elif args.phase == 'adapt':
        dict_pretrain_model_id = dict(list(dict_be_pretrain.items()) + list(dict_cbwdrop0.items()))
    elif args.phase == 'retrain':
        dict_pretrain_model_id = dict(
            list(dict_be_pretrain.items()) + list(dict_cbwdrop0.items()) + list(dict_cbwdrop1.items()))
    else:
        dict_pretrain_model_id = None

    # Add info for Delta Networks
    if 'Delta' in args.hid_type or args.hid_type in ['dLSTM']:
        dict_model_id = dict(list(dict_model_id.items()) + list(dict_deltanet.items()))

    # Model ID
    list_model_id = []
    for item in list(dict_model_id.items()):
        list_model_id += list(item)
    model_id = '_'.join(list_model_id)

    # Pretrain Model ID
    if dict_pretrain_model_id is not None:
        list_pretrain_model_id = []
        for item in list(dict_pretrain_model_id.items()):
            list_pretrain_model_id += list(item)
        pretrain_model_id = '_'.join(list_pretrain_model_id)
    else:
        pretrain_model_id = None

    return model_id, pretrain_model_id


def write_log(args, logger, tb_writer, model_id, train_stat, val_stat, test_stat, net, optimizer, epoch,
              time_curr,
              alpha, retrain):
    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    # # Evaluate Weight Range
    # for name, param in net.named_parameters():
    #     param_data = param.data
    #     print("Param Name: %30s | Min: %f | Max: %f" % (name, torch.min(param_data), torch.max(param_data)))
    #
    # # Evaluate Gradient Range
    # for name, param in net.named_parameters():
    #     param_data_grad = param.grad
    #     print("Grad Name: %30s | Min: %f | Max: %f" % (name, torch.min(param_data_grad), torch.max(param_data_grad)))

    # Evaluate RNN Weight Sparsity
    n_nonzero_weight_elem = 0
    n_weight_elem = 0
    for name, param in net.named_parameters():
        if 'rnn' in name:
            if 'weight' in name:
                n_nonzero_weight_elem += len(param.data.nonzero())
                n_weight_elem += param.data.nelement()
    sp_w_rnn = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Evaluate FC Layer Weight Sparsity
    sp_w_fc = 0
    if args.fc_extra_size:
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in net.named_parameters():
            if 'fc_extra' in name:
                if 'weight' in name:
                    n_nonzero_weight_elem += len(param.data.nonzero())
                    n_weight_elem += param.data.nelement()
        sp_w_fc = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    # Create Log List
    list_log_headers = ['EPOCH', 'TIME', 'SEED', 'N_PARAM', 'BATCH_SIZE',
                        'LR_END', 'DECAY_FACTOR', 'PATIENCE', 'CLA_TYPE',
                        'CLA_SIZE', 'QA', 'AQI', 'AQF', 'QW', 'WQI', 'WQF',
                        'QG', 'GQI', 'GQF', 'SP_W_RNN', 'SP_W_FC', 'TH_X',
                        'TH_H', 'LAMBDA', 'CLIP_GRAD_NORM_MAX', 'NUM_ARRAY_PE', 'CBWDROP',
                        'ALPHA', 'ALPHA_ANNEAL_EPOCH', 'GAMMA_RNN', 'GAMMA_FC', 'WEIGHT_DECAY', 'HID_DROPOUT']
    list_log_values = [epoch + 1, time_curr, args.seed, n_param, args.batch_size,
                       args.lr_end, args.decay_factor, args.patience, args.hid_type,
                       args.hid_size, args.qa, args.aqi, args.aqf, args.qw, args.wqi, args.wqf,
                       args.qg, args.gqi, args.gqf, sp_w_rnn, sp_w_fc, args.th_x,
                       args.th_h, args.beta, args.clip_grad_norm_max, args.num_array_pe, args.cbtd,
                       alpha, args.alpha_anneal_epoch, args.gamma_rnn, args.gamma_fc, args.weight_decay,
                       args.hid_dropout]
    if optimizer is not None:
        list_log_headers.append('LR')
        list_log_values.append(lr_curr)
    if train_stat is not None:
        list_log_headers.extend(['LOSS_TRAIN', 'REG_TRAIN'])
        list_log_values.extend([train_stat['loss'], train_stat['reg']])
    if val_stat is not None:
        list_log_headers.append('LOSS_VAL')
        list_log_values.append(val_stat['loss'])
        if args.score_val:
            list_log_headers.extend(['PER_VAL', 'SP_DX_VAL', 'SP_DH_VAL'])
            list_log_values.extend([val_stat['per'], val_stat['sp_dx'], val_stat['sp_dh']])
    if test_stat is not None:
        list_log_headers.append('LOSS_TEST')
        list_log_values.append(test_stat['loss'])
        if args.score_test:
            list_log_headers.extend(['PER_TEST', 'SP_DX_TEST', 'SP_DH_TEST'])
            list_log_values.extend([test_stat['per'], test_stat['sp_dx'], test_stat['sp_dh']])

    # Write Log
    logger.add_row(list_log_headers, list_log_values)
    logger.write_csv()

    # Tensorboard
    if tb_writer is not None:
        tb_writer.add_scalars(model_id, {'L-Train': train_stat['loss'],
                                         'L-Val': val_stat['loss']}, epoch)


def print_log(args, log_stat, train_stat, val_stat, test_stat):
    str_print = f"Epoch: {log_stat['EPOCH'] + 1:3d} of {log_stat['N_EPOCH']:3d} " \
                f"| Time: {log_stat['TIME_CURR']:s} " \
                f"| LR: {log_stat['LR_CURR']:1.5f} " \
                f"| Alpha: {log_stat['ALPHA']:1.2f} " \
                f"| Sp.W {log_stat['SP_W_RNN'] * 100:3.2f}% " \
                f"| Sp.Wfc {log_stat['SP_W_FC'] * 100:3.2f}% |\n"

    # Print train set stats
    if train_stat is not None:
        str_print += f"    | Train-Loss: {log_stat['TRAIN_LOSS']:4.3f} " \
                     f"| Train-Reg: {log_stat['TRAIN_REG']:4.2f} |\n"

    # Print val set stats
    if val_stat is not None:
        str_print += f"    | Val-Loss: {log_stat['VAL_LOSS']:4.3f}"
        if args.score_val:
            str_print += f" | Val-PER: {log_stat['VAL_PER'] * 100:3.3f}% "
            if 'VAL_SP_DX' in log_stat.keys():
                str_print += f"| Val-Sp-dx: {log_stat['VAL_SP_DX'] * 100:3.2f}% "
            if 'VAL_SP_DH' in log_stat.keys():
                str_print += f"| Val-Sp-dh {log_stat['VAL_SP_DH'] * 100:3.2f}% |"
        str_print += '\n'

    # Print test set stats
    if test_stat is not None:
        str_print += f"    | Test-Loss: {log_stat['TEST_LOSS']:4.3f}"
        if args.score_test:
            str_print += f" | Test-PER: {log_stat['TEST_PER'] * 100:3.2f}% "
            if 'TEST_SP_DX' in log_stat.keys():
                str_print += f"| Test-Sp-dx: {log_stat['TEST_SP_DX'] * 100:3.2f} "
            if 'TEST_SP_DH' in log_stat.keys():
                str_print += f"| Test-Sp-dh: {log_stat['TEST_SP_DH'] * 100:3.2f} |"
    print(str_print)


def save_best_model(args,
                    best_val,
                    net,
                    save_file,
                    logger,
                    logfile_best,
                    epoch,
                    val_stat,
                    score_val):
    # Save best model according to the best score on the validation set
    if score_val:
        if epoch == 0:
            best_val = val_stat['per']
            torch.save(net.state_dict(), save_file)
            best_epoch = epoch
            logger.write_log_idx(best_epoch, logfile_best)
            print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
        if val_stat['per'] <= best_val:
            best_val = val_stat['per']
            best_epoch = epoch
            logger.write_log_idx(best_epoch, logfile_best)
            torch.save(net.state_dict(), save_file)
            print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
    # Save best model according to the lowest loss on the validation set
    else:
        if epoch == 0:
            best_val = val_stat['loss']
            torch.save(net.state_dict(), save_file)
            best_epoch = epoch
            logger.write_log_idx(best_epoch, logfile_best)
            print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
        if val_stat['loss'] <= best_val:
            best_val = val_stat['loss']
            best_epoch = epoch
            # Record the best epoch
            logger.write_log_idx(best_epoch, logfile_best)
            torch.save(net.state_dict(), save_file)
            print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
    return best_val


def print_training_header(args, n_features, n_classes, n_param):
    """Print training header with model info."""
    if RICH_AVAILABLE:
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
    else:
        print("=" * 60)
        print(f"Dataset: {args.dataset_name}")
        print(f"Model Type: {args.hid_type}")
        print(f"Hidden: {args.hid_layers} layers x {args.hid_size} units")
        print(f"Input Features: {n_features}, Output Classes: {n_classes}")
        print(f"Total Parameters: {n_param:,}")
        print(f"Batch Size: {args.batch_size}, LR: {args.lr}, Epochs: {args.n_epochs}")
        print("=" * 60)


def print_model_analysis(net, args, n_features, seq_len=777, device='cpu'):
    """
    Print detailed model analysis using torchstat2-based model stats.

    Args:
        net: PyTorch model
        args: Arguments namespace
        n_features: Number of input features
        seq_len: Sequence length for RNN models (TIMIT avg ~777 frames)
        device: Device to run analysis on
    """
    if not MODEL_STATS_AVAILABLE:
        print("Model stats not available. Skipping detailed analysis.")
        return None

    try:
        model_name = f"{args.hid_type} ({args.hid_layers}L x {args.hid_size}H)"
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
        print(f"Model analysis failed: {e}")
        return None


def print_sparsity_analysis(net, args):
    """Print detailed sparsity analysis for the model."""
    if not RICH_AVAILABLE:
        # Plain text version
        print("\n" + "=" * 60)
        print("WEIGHT SPARSITY ANALYSIS")
        print("=" * 60)

        total_params = 0
        total_nonzero = 0

        for name, param in net.named_parameters():
            if 'weight' in name:
                n_total = param.data.numel()
                n_nonzero = torch.count_nonzero(param.data).item()
                sparsity = 1 - (n_nonzero / n_total) if n_total > 0 else 0
                total_params += n_total
                total_nonzero += n_nonzero
                print(f"  {name[:35]:<35} {n_total:>10,} {n_nonzero:>10,} {sparsity*100:>6.1f}%")

        overall_sparsity = 1 - (total_nonzero / total_params) if total_params > 0 else 0
        print("-" * 60)
        print(f"  {'TOTAL':<35} {total_params:>10,} {total_nonzero:>10,} {overall_sparsity*100:>6.1f}%")
        return

    # Rich version
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

        if not RICH_AVAILABLE:
            print("\n" + "=" * 60)
            print("TEMPORAL SPARSITY STATISTICS")
            print("=" * 60)
            print(f"  Input Delta Sparsity (Δx):   {sp_dx*100:.2f}%")
            print(f"  Hidden Delta Sparsity (Δh):  {sp_dh*100:.2f}%")
            print(f"  Combined Sparsity (Δxh):     {sp_dxh*100:.2f}%")
            return

        console.print()
        console.print(Panel("[bold orange1]Temporal Sparsity Statistics[/]", border_style="orange1"))

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
        print(f"Failed to get delta stats: {e}")
