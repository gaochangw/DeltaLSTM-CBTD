"""
Ablation Study: MSQ (LSB Regularization) + BitWave (Sign-Magnitude Encoding)
for bit-level sparsity enhancement in DeltaLSTM.

Configurations tested:
  A. Baseline          - Standard retrain (CBTD only), two's complement
  B. BitWave only      - Standard retrain, sign-magnitude bit counting
  C. MSQ (k=2, low)    - Retrain with LSB penalty lambda=0.01, two's complement
  D. MSQ (k=2, med)    - Retrain with LSB penalty lambda=0.1, two's complement
  E. MSQ (k=2, high)   - Retrain with LSB penalty lambda=1.0, two's complement
  F. MSQ (k=3, med)    - Retrain with LSB penalty on 3 LSBs lambda=0.1, two's complement
  G. MSQ + BitWave     - Best MSQ config + sign-magnitude bit counting

Usage:
  python ablation_bit_sparsity.py [--n_epochs 2] [--use_cuda 0]
"""

import os
import sys
import copy
import time
import math
import argparse
import random as rnd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import importlib

import model as model
from utils import util
from utils.util import gen_paths, count_net_params
from utils.msq_regularizer import compute_lsb_penalty
from utils.sparsity_analysis import (
    analyze_sparsity_scenarios,
    compute_bit_level_sparsity,
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    console = Console()
    RICH = True
except ImportError:
    RICH = False


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation: MSQ + BitWave bit-level sparsity')
    parser.add_argument('--n_epochs', default=2, type=int, help='Retrain epochs per config')
    parser.add_argument('--use_cuda', default=0, type=int, help='Use GPU')
    parser.add_argument('--gpu_device', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    return parser.parse_args()


def load_gscdv2_args():
    """Load default GSCDv2 arguments."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    module_args = importlib.import_module('modules.gscdv2.arguments')
    parent_parser = module_args.add_args(parent_parser)
    args = parent_parser.parse_args([])
    args.dataset_name = 'gscdv2'
    # Set file paths
    args.trainfile = 'feat/gscdv2/TRAIN_D_gscdv2_NF_40_SI_0.025_ST_0.01_GC_0_FG_1.h5'
    args.valfile = 'feat/gscdv2/VAL_D_gscdv2_NF_40_SI_0.025_ST_0.01_GC_0_FG_1.h5'
    args.testfile = 'feat/gscdv2/TEST_D_gscdv2_NF_40_SI_0.025_ST_0.01_GC_0_FG_1.h5'
    return args


def setup_device(args, ablation_args):
    """Configure compute device."""
    args.use_cuda = ablation_args.use_cuda
    args.gpu_device = ablation_args.gpu_device
    if torch.cuda.is_available() and args.use_cuda:
        torch.cuda.set_device(args.gpu_device)
        args.device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        args.device = torch.device('mps')
    else:
        args.device = torch.device('cpu')
    return args


def create_deltalstm_model(args, n_features, n_classes, pretrain_path):
    """Create a DeltaLSTM model and load pretrained LSTM weights."""
    args.hid_type = 'DeltaLSTM'
    net = model.Model(args=args, input_size=n_features, n_classes=n_classes)
    net = util.load_model(net, pretrain_path, args.use_cuda, args.gpu_device)
    net = net.to(args.device)
    return net


def retrain_with_msq(args, net, dataloader, n_epochs, msq_lambda=0.0, msq_k=2, lr=1e-3):
    """
    Retrain DeltaLSTM with optional MSQ LSB regularization.

    Returns: (net, test_stat) after retrain + final eval.
    """
    module_train_func = importlib.import_module('modules.gscdv2.train_func')
    get_batch_data = module_train_func.get_batch_data
    calculate_loss = module_train_func.calculate_loss
    forward_propagation = module_train_func.forward_propagation
    process_network = module_train_func.process_network

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(n_epochs):
        # Process network (quantize + CBTD)
        alpha = 1.0  # Retrain phase: alpha=1
        net, net_for_eval = process_network(args, net=net, alpha=alpha)

        # Train
        net.train()
        net.set_qa_fc_final(0)
        epoch_loss = 0
        n_batches = 0

        for dict_batch_array in tqdm(
            dataloader.iterate(epoch=epoch, set_name='train', batch_size=args.batch_size,
                               mode='batch', shuffle_type='random', enable_gauss=0),
            desc=f'  Train E{epoch+1}', unit='b',
            total=dataloader.get_num_batch('train', args.batch_size)
        ):
            dict_batch_tensor = get_batch_data(args, dict_batch_array)
            optimizer.zero_grad()

            net_out, _, reg = forward_propagation(net, dict_batch_tensor)
            loss, _ = calculate_loss(criterion, net_out, dict_batch_tensor, reg, args.beta)

            # Add MSQ LSB penalty
            if msq_lambda > 0:
                lsb_pen = compute_lsb_penalty(net, qf=args.wqf, k=msq_k, target_layers='rnn')
                loss = loss + msq_lambda * lsb_pen

            loss.backward()
            if args.clip_grad_norm_max != 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm_max)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            del dict_batch_tensor, dict_batch_array, loss, net_out

        avg_loss = epoch_loss / n_batches
        print(f'    Epoch {epoch+1}: loss={avg_loss:.4f}')

    # Final quantize + CBTD
    net, net_for_eval = process_network(args, net=net, alpha=1.0)

    # Evaluate
    test_stat = evaluate_model(args, net_for_eval, dataloader)
    return net_for_eval, test_stat


def evaluate_model(args, net, dataloader):
    """Run evaluation on test set, return stat dict with accuracy + temporal sparsity."""
    module_train_func = importlib.import_module('modules.gscdv2.train_func')
    module_metric = importlib.import_module('modules.gscdv2.metric')
    get_batch_data = module_train_func.get_batch_data
    calculate_loss = module_train_func.calculate_loss
    forward_propagation = module_train_func.forward_propagation
    add_meter_data = module_train_func.add_meter_data
    Meter = module_metric.Meter
    gen_meter_args = module_metric.gen_meter_args

    criterion = nn.CrossEntropyLoss()
    n_classes = 12
    dict_meter_args = gen_meter_args(args, n_classes)
    meter = Meter(dict_meter_args)

    net.eval()
    net.set_qa_fc_final(args.qa_fc_final)
    epoch_loss = 0
    n_batches = 0
    dict_meter_data = {'net_out': [], 'net_qout': []}

    with torch.no_grad():
        for dict_batch_array in tqdm(
            dataloader.iterate(epoch=0, set_name='test', batch_size=args.batch_size_eval,
                               mode='batch', shuffle_type='high_throughput', enable_gauss=0),
            desc='  Eval', unit='b',
            total=dataloader.get_num_batch('test', args.batch_size_eval)
        ):
            dict_batch_tensor = get_batch_data(args, dict_batch_array)
            net_out, _, reg = forward_propagation(net, dict_batch_tensor)
            loss, _ = calculate_loss(criterion, net_out, dict_batch_tensor, reg, args.beta)
            epoch_loss += loss.item()
            n_batches += 1

            dict_meter_data['net_out'].append(net_out.detach().cpu())
            for k, v in dict_batch_tensor.items():
                if k == 'features':
                    continue
                try:
                    dict_meter_data[k].append(v.detach().cpu())
                except KeyError:
                    dict_meter_data[k] = [v.detach().cpu()]
            del dict_batch_tensor, dict_batch_array, loss, net_out

    epoch_loss /= n_batches
    meter = add_meter_data(args, meter, dict_meter_data)
    stat = {'loss': epoch_loss, 'lr_criterion': epoch_loss}

    # Temporal sparsity
    if hasattr(net, 'rnn') and hasattr(net.rnn, 'get_temporal_sparsity'):
        dict_stats = net.rnn.get_temporal_sparsity()
        stat['sp_dx'] = dict_stats['sparsity_delta_x']
        stat['sp_dh'] = dict_stats['sparsity_delta_h']
        stat['sp_dxh'] = dict_stats['sparsity_delta']

    # Accuracy
    stat = meter.get_metrics(stat)
    return stat


def get_sparsity_metrics(net, args, test_stat, representation='twos_complement'):
    """Extract key sparsity metrics for the ablation table."""
    analysis = analyze_sparsity_scenarios(net, args, temporal_stats=test_stat, representation=representation)
    agg = analysis['aggregate']
    s5 = analysis['scenarios'][4]  # Scenario 5
    return {
        'sp_w': agg['element_sparsity'],
        'bit_sp_all': agg['bit_sparsity'],
        'bit_sp_nz': agg['bit_sparsity_nonzero'],
        'avg_nz_bits': agg['avg_nonzero_bits_per_nz_element'],
        'sp_dxh': analysis['temporal']['sp_dxh'],
        'speedup_s3': analysis['scenarios'][2]['speedup'],
        'speedup_s5': s5['speedup'],
    }


def print_ablation_table(results):
    """Print the ablation comparison table."""
    if RICH:
        _print_rich_ablation(results)
    else:
        _print_plain_ablation(results)


def _print_rich_ablation(results):
    console.print()
    console.print(Panel(
        "[bold bright_cyan]Ablation Study: MSQ + BitWave Bit-Level Sparsity[/]",
        border_style="bright_cyan"))

    table = Table(
        title="Ablation Results (GSCDv2 DeltaLSTM, Q1.7, gamma=0.94)",
        box=box.DOUBLE_EDGE, show_header=True, header_style="bold white"
    )
    table.add_column("Config", style="bold", width=28)
    table.add_column("Acc %", justify="right", width=8)
    table.add_column("sp_w %", justify="right", width=8)
    table.add_column("bit_sp\n(all) %", justify="right", width=9)
    table.add_column("bit_sp\n(nz) %", justify="right", width=9)
    table.add_column("NZ bits\n/nz elem", justify="right", width=9)
    table.add_column("sp_dxh %", justify="right", width=9)
    table.add_column("S3\nSpeedup", justify="right", width=8)
    table.add_column("S5\nSpeedup", justify="right", width=9)

    # Find best S5 speedup for highlighting
    best_s5 = max(r['metrics']['speedup_s5'] for r in results)

    for r in results:
        m = r['metrics']
        acc = r['accuracy'] * 100
        is_best_s5 = abs(m['speedup_s5'] - best_s5) < 0.01

        s5_str = f"{m['speedup_s5']:.1f}x"
        if is_best_s5:
            s5_str = f"[bold bright_green]{s5_str}[/]"

        table.add_row(
            r['name'],
            f"{acc:.2f}",
            f"{m['sp_w']*100:.1f}",
            f"{m['bit_sp_all']*100:.1f}",
            f"{m['bit_sp_nz']*100:.1f}",
            f"{m['avg_nz_bits']:.2f}/8",
            f"{m['sp_dxh']*100:.1f}",
            f"{m['speedup_s3']:.1f}x",
            s5_str,
        )

    console.print(table)

    # Delta table: improvement over baseline
    baseline = results[0]['metrics']
    baseline_acc = results[0]['accuracy']

    console.print()
    delta_table = Table(
        title="Delta vs Baseline",
        box=box.ROUNDED, show_header=True, header_style="bold yellow"
    )
    delta_table.add_column("Config", style="bold", width=28)
    delta_table.add_column("Acc delta", justify="right", width=10)
    delta_table.add_column("bit_sp(nz)\ndelta", justify="right", width=12)
    delta_table.add_column("S5 Speedup\ndelta", justify="right", width=12)

    for r in results[1:]:
        m = r['metrics']
        acc_delta = (r['accuracy'] - baseline_acc) * 100
        bit_delta = (m['bit_sp_nz'] - baseline['bit_sp_nz']) * 100
        s5_delta = m['speedup_s5'] - baseline['speedup_s5']

        acc_color = "green" if acc_delta >= 0 else "red"
        bit_color = "green" if bit_delta > 0 else ("red" if bit_delta < 0 else "white")
        s5_color = "green" if s5_delta > 0 else ("red" if s5_delta < 0 else "white")

        delta_table.add_row(
            r['name'],
            f"[{acc_color}]{acc_delta:+.2f}pp[/]",
            f"[{bit_color}]{bit_delta:+.1f}pp[/]",
            f"[{s5_color}]{s5_delta:+.1f}x[/]",
        )

    console.print(delta_table)
    console.print()


def _print_plain_ablation(results):
    print()
    print("=" * 110)
    print("  Ablation Study: MSQ + BitWave Bit-Level Sparsity")
    print("=" * 110)
    header = (f"{'Config':<28} {'Acc%':>7} {'sp_w%':>7} {'bit_sp%':>8} "
              f"{'bit_nz%':>8} {'NZbits':>7} {'dxh%':>7} {'S3':>7} {'S5':>8}")
    print(header)
    print("-" * 110)
    for r in results:
        m = r['metrics']
        print(f"{r['name']:<28} {r['accuracy']*100:>7.2f} {m['sp_w']*100:>7.1f} "
              f"{m['bit_sp_all']*100:>8.1f} {m['bit_sp_nz']*100:>8.1f} "
              f"{m['avg_nz_bits']:>5.2f}/8 {m['sp_dxh']*100:>7.1f} "
              f"{m['speedup_s3']:>6.1f}x {m['speedup_s5']:>7.1f}x")
    print("=" * 110)


def main():
    ablation_args = parse_args()

    print("=" * 70)
    print("  Ablation Study: MSQ + BitWave for Bit-Level Sparsity")
    print("=" * 70)

    # Setup
    args = load_gscdv2_args()
    args = setup_device(args, ablation_args)
    args.batch_size = ablation_args.batch_size
    args.batch_size_eval = ablation_args.batch_size_eval

    # Reproducibility
    rnd.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load data
    module_dataloader = importlib.import_module('modules.gscdv2.dataloader')
    CustomDataLoader = module_dataloader.CustomDataLoader
    dataloader = CustomDataLoader(
        trainfile=args.trainfile, valfile=args.valfile, testfile=args.testfile,
        args=args, qf=None, log_feat=None, normalization=args.normalization
    )
    n_features = dataloader.n_features
    n_classes = dataloader.n_classes
    print(f"Features: {n_features}, Classes: {n_classes}")

    # Find pretrained model
    pretrain_model_id = (
        'SD_4_L_1_IN_123_H_128_CLA_LSTM_FC_0_QA_1_AI_8_AF_8_QW_1_WI_1_WF_7'
        '_CBWD_1_SPRNN_0.94_SPFC_0.94_AAE_30_NAP_4'
    )
    pretrain_path = os.path.join('./save/gscdv2/pretrain', pretrain_model_id + '.pt')
    if not os.path.exists(pretrain_path):
        print(f"ERROR: Pretrained model not found: {pretrain_path}")
        sys.exit(1)
    print(f"Pretrained model: {pretrain_path}")

    # Also check for existing retrained model (use as baseline if available)
    retrain_model_path = (
        './save/gscdv2/retrain/SD_4_L_1_IN_123_H_128_CLA_DeltaLSTM_FC_0_QA_1_AI_8_AF_8_QW_1_WI_1_WF_7'
        '_CBWD_1_SPRNN_0.94_SPFC_0.94_AAE_30_NAP_4_TX_0.50_TH_0.50_BT_0.0e+00.pt'
    )

    n_epochs = ablation_args.n_epochs
    lr = ablation_args.lr

    # =========================================================================
    # Define ablation configurations
    # =========================================================================
    configs = [
        {'name': 'A: Baseline (TC)',       'msq_lambda': 0.0, 'msq_k': 2, 'rep': 'twos_complement'},
        {'name': 'B: BitWave only (SM)',    'msq_lambda': 0.0, 'msq_k': 2, 'rep': 'sign_magnitude',
         'skip_retrain': True, 'share_model': 'A: Baseline (TC)'},
        {'name': 'C: MSQ k=2 lam=0.01',    'msq_lambda': 0.01, 'msq_k': 2, 'rep': 'twos_complement'},
        {'name': 'D: MSQ k=2 lam=0.1',     'msq_lambda': 0.1,  'msq_k': 2, 'rep': 'twos_complement'},
        {'name': 'E: MSQ k=2 lam=1.0',     'msq_lambda': 1.0,  'msq_k': 2, 'rep': 'twos_complement'},
        {'name': 'F: MSQ k=3 lam=0.1',     'msq_lambda': 0.1,  'msq_k': 3, 'rep': 'twos_complement'},
    ]

    # =========================================================================
    # Run ablation
    # =========================================================================
    all_results = []
    model_cache = {}  # Cache retrained models for shared configs

    for i, cfg in enumerate(configs):
        print()
        print(f"--- [{i+1}/{len(configs)}] {cfg['name']} ---")

        if cfg.get('skip_retrain') and cfg.get('share_model'):
            # Reuse model from another config, just re-analyze with different representation
            shared = cfg['share_model']
            print(f"  Reusing model from '{shared}', re-analyzing with {cfg['rep']}")
            net = model_cache[shared]['net']
            test_stat = model_cache[shared]['test_stat']
        else:
            # Set seeds for reproducibility
            rnd.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            # Set DeltaLSTM args
            args.hid_type = 'DeltaLSTM'
            args.th_x = 0.50
            args.th_h = 0.50
            args.phase = 'retrain'

            # Create fresh model from pretrain
            net = create_deltalstm_model(args, n_features, n_classes, pretrain_path)
            print(f"  Retraining {n_epochs} epochs, MSQ lambda={cfg['msq_lambda']}, k={cfg['msq_k']}")

            net, test_stat = retrain_with_msq(
                args, net, dataloader,
                n_epochs=n_epochs,
                msq_lambda=cfg['msq_lambda'],
                msq_k=cfg['msq_k'],
                lr=lr,
            )
            model_cache[cfg['name']] = {'net': net, 'test_stat': test_stat}

        # Collect metrics
        accuracy = test_stat.get('accuracy', 0)
        metrics = get_sparsity_metrics(net, args, test_stat, representation=cfg['rep'])

        result = {
            'name': cfg['name'],
            'accuracy': accuracy,
            'loss': test_stat.get('loss', 0),
            'metrics': metrics,
            'config': cfg,
        }
        all_results.append(result)
        print(f"  Acc={accuracy*100:.2f}%  bit_sp_nz={metrics['bit_sp_nz']*100:.1f}%  "
              f"S5={metrics['speedup_s5']:.1f}x")

    # Add combined MSQ+BitWave configs (best MSQ + sign-magnitude)
    # Find best MSQ config by S5 speedup among TC configs (excluding baseline)
    msq_results = [r for r in all_results if r['config']['msq_lambda'] > 0]
    if msq_results:
        # Pick MSQ config with best S5 speedup
        best_msq = max(msq_results, key=lambda r: r['metrics']['speedup_s5'])
        best_name = best_msq['name']
        print()
        print(f"--- [Combined] Best MSQ ({best_name}) + BitWave ---")
        cached = model_cache[best_name]
        metrics_sm = get_sparsity_metrics(
            cached['net'], args, cached['test_stat'], representation='sign_magnitude')
        combined_result = {
            'name': f'G: {best_name[3:]} + SM',
            'accuracy': best_msq['accuracy'],
            'loss': best_msq['loss'],
            'metrics': metrics_sm,
            'config': {**best_msq['config'], 'rep': 'sign_magnitude'},
        }
        all_results.append(combined_result)
        print(f"  Acc={best_msq['accuracy']*100:.2f}%  bit_sp_nz={metrics_sm['bit_sp_nz']*100:.1f}%  "
              f"S5={metrics_sm['speedup_s5']:.1f}x")

    # =========================================================================
    # Print summary table
    # =========================================================================
    print_ablation_table(all_results)


if __name__ == '__main__':
    main()
