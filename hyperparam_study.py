"""
Hyperparameter Study for DeltaLSTM-CBTD

Explores the impact of:
- Target sparsity (gamma_rnn): 0% to 95%
- Delta threshold (th_x = th_h): 0 to 0.5

On metrics:
- Accuracy, F1 score, AUC-ROC
- Model size (#parameters, #non-zero parameters)
- FLOPs (dense vs sparse)
- Temporal sparsity

Outputs:
- Excel file with detailed results (multiple sheets)
- Beautiful plots summarizing the study
"""

import os
import sys
import argparse
import time
import json
import random as rnd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CTCLoss
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Import project modules
import model as net_model
from utils import util
from utils.util import gen_paths, count_net_params
import importlib


class HyperparamStudy:
    """Hyperparameter study manager."""

    def __init__(self, args):
        self.args = args
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Output directories
        self.output_dir = f"./study_results/{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)

        # Load dataset modules
        self.load_modules()

    def load_modules(self):
        """Load dataset-specific modules."""
        dataset_name = self.args.dataset_name
        self.module_log = importlib.import_module(f'modules.{dataset_name}.log')
        self.module_dataloader = importlib.import_module(f'modules.{dataset_name}.dataloader')
        self.module_train_func = importlib.import_module(f'modules.{dataset_name}.train_func')
        self.module_metric = importlib.import_module(f'modules.{dataset_name}.metric')

    def setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available() and self.args.use_cuda:
            torch.cuda.set_device(self.args.gpu_device)
            self.args.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(self.args.gpu_device)}")
        else:
            self.args.device = torch.device('cpu')
            print("Using CPU")

    def setup_reproducibility(self, seed=42):
        """Setup reproducibility."""
        rnd.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def create_dataloader(self):
        """Create dataloader."""
        CustomDataLoader = self.module_dataloader.CustomDataLoader
        dataloader = CustomDataLoader(
            trainfile=self.args.trainfile,
            valfile=self.args.valfile,
            testfile=self.args.testfile,
            args=self.args,
            qf=None,
            log_feat=None,
            normalization=self.args.normalization
        )
        return dataloader

    def count_flops(self, net, seq_len=98, batch_size=1):
        """Estimate FLOPs for the model."""
        # LSTM FLOPs per timestep: 4 * (input_size * hidden_size + hidden_size * hidden_size) * 2
        # (multiply-accumulate = 2 ops)
        input_size = net.inp_size
        hidden_size = net.hid_size
        num_layers = net.hid_layers
        n_classes = 12

        # RNN FLOPs per timestep per layer
        lstm_flops_per_step = 4 * (input_size * hidden_size + hidden_size * hidden_size) * 2

        # For layers > 1, input is hidden_size
        for layer in range(1, num_layers):
            lstm_flops_per_step += 4 * (hidden_size * hidden_size + hidden_size * hidden_size) * 2

        # Total RNN FLOPs
        rnn_flops = lstm_flops_per_step * seq_len * batch_size

        # FC layer FLOPs
        if net.fc_extra_size:
            fc_flops = (hidden_size * net.fc_extra_size + net.fc_extra_size * n_classes) * 2 * batch_size
        else:
            fc_flops = hidden_size * n_classes * 2 * batch_size

        return rnn_flops + fc_flops

    def count_sparse_flops(self, net, temporal_sparsity, weight_sparsity, seq_len=98):
        """Estimate sparse FLOPs considering temporal and weight sparsity."""
        dense_flops = self.count_flops(net, seq_len)

        # Sparse FLOPs = Dense FLOPs * (1 - temporal_sparsity) * (1 - weight_sparsity)
        sparse_flops = dense_flops * (1 - temporal_sparsity) * (1 - weight_sparsity)

        return sparse_flops

    def train_epoch(self, net, optimizer, criterion, dataloader, args):
        """Train for one epoch."""
        net.train()
        net.set_qa_fc_final(0)

        get_batch_data = self.module_train_func.get_batch_data
        forward_propagation = self.module_train_func.forward_propagation
        calculate_loss = self.module_train_func.calculate_loss

        epoch_loss = 0
        n_batches = 0

        for dict_batch_array in dataloader.iterate(
            epoch=0, set_name='train', batch_size=args.batch_size,
            mode='batch', shuffle_type='random', enable_gauss=0
        ):
            dict_batch_tensor = get_batch_data(args, dict_batch_array)
            optimizer.zero_grad()
            net_out, _, reg = forward_propagation(net, dict_batch_tensor)
            loss, _ = calculate_loss(criterion, net_out, dict_batch_tensor, reg, args.beta)
            loss.backward()

            if args.clip_grad_norm_max != 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm_max)

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

            del dict_batch_tensor, loss, net_out

        return epoch_loss / n_batches

    def evaluate(self, net, criterion, dataloader, args, set_name='test'):
        """Evaluate model."""
        net.eval()
        net.set_qa_fc_final(args.qa_fc_final)

        get_batch_data = self.module_train_func.get_batch_data
        forward_propagation = self.module_train_func.forward_propagation
        calculate_loss = self.module_train_func.calculate_loss
        add_meter_data = self.module_train_func.add_meter_data
        Meter = self.module_metric.Meter
        gen_meter_args = self.module_metric.gen_meter_args

        dict_meter_args = gen_meter_args(args, dataloader.n_classes)
        meter = Meter(dict_meter_args)

        epoch_loss = 0
        n_batches = 0
        dict_meter_data = {'net_out': [], 'targets_metric': []}

        with torch.no_grad():
            for dict_batch_array in dataloader.iterate(
                epoch=0, set_name=set_name, batch_size=args.batch_size_eval,
                mode='batch', shuffle_type='high_throughput', enable_gauss=0
            ):
                dict_batch_tensor = get_batch_data(args, dict_batch_array)
                net_out, _, reg = forward_propagation(net, dict_batch_tensor)
                loss, _ = calculate_loss(criterion, net_out, dict_batch_tensor, reg, args.beta)

                epoch_loss += loss.item()
                n_batches += 1

                dict_meter_data['net_out'].append(net_out.detach().cpu())
                dict_meter_data['targets_metric'].append(dict_batch_tensor['targets_metric'].detach().cpu())

                del dict_batch_tensor, loss, net_out

        # Get metrics
        meter = add_meter_data(args, meter, dict_meter_data)
        stat = {'loss': epoch_loss / n_batches, 'lr_criterion': epoch_loss / n_batches}
        stat = meter.get_metrics(stat)

        # Get DeltaRNN stats if applicable
        if "Delta" in args.hid_type:
            try:
                dict_stats = net.rnn.get_temporal_sparsity()
                stat['sp_dx'] = dict_stats['sparsity_delta_x']
                stat['sp_dh'] = dict_stats['sparsity_delta_h']
                stat['sp_delta'] = dict_stats['sparsity_delta']
            except:
                stat['sp_dx'] = 0
                stat['sp_dh'] = 0
                stat['sp_delta'] = 0

        return stat

    def process_network(self, net, args, alpha):
        """Process network for quantization and CBTD."""
        for name, param in net.named_parameters():
            param.data = util.quantize_tensor(param.data, args.wqi, args.wqf, args.qw)

            if args.cbtd:
                if 'rnn' in name and 'weight' in name:
                    util.cbtd(param.data, gamma=args.gamma_rnn, alpha=alpha,
                              balance_pe=args.balance_pe, num_pe=args.num_array_pe)
                if 'fc_extra' in name and 'weight' in name:
                    util.cbtd(param.data, gamma=args.gamma_fc, alpha=alpha,
                              balance_pe=args.balance_pe, num_pe=args.num_array_pe)
        return net

    def get_weight_sparsity(self, net):
        """Calculate weight sparsity."""
        n_nonzero = 0
        n_total = 0
        for name, param in net.named_parameters():
            if 'rnn' in name and 'weight' in name:
                n_nonzero += torch.count_nonzero(param.data).item()
                n_total += param.data.numel()

        return 1 - (n_nonzero / n_total) if n_total > 0 else 0

    def run_single_experiment(self, gamma_rnn, threshold, pretrain_epochs=50, retrain_epochs=5):
        """Run a single experiment with given hyperparameters."""
        args = argparse.Namespace(**vars(self.args))

        # Set hyperparameters
        args.gamma_rnn = gamma_rnn
        args.gamma_fc = gamma_rnn
        args.th_x = threshold
        args.th_h = threshold

        self.setup_reproducibility(args.seed)

        # Create dataloader
        dataloader = self.create_dataloader()
        n_features = dataloader.n_features
        n_classes = dataloader.n_classes

        # Loss function
        criterion = nn.CrossEntropyLoss()

        result = {
            'gamma_rnn': gamma_rnn,
            'threshold': threshold,
            'n_features': n_features,
            'n_classes': n_classes,
        }

        # ==================
        # Phase 1: Pretrain LSTM
        # ==================
        args.hid_type = 'LSTM'
        args.phase = 'pretrain'

        net = net_model.Model(args=args, input_size=n_features, n_classes=n_classes)
        net = net.to(args.device)

        # Initialize
        self.module_train_func.initialize_network(net, args)

        # Count parameters
        n_params = count_net_params(net)
        result['n_params_total'] = n_params

        # Optimizer
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.decay_factor, patience=args.patience, min_lr=args.lr_end
        )

        print(f"\n{'='*60}")
        print(f"Pretrain LSTM: gamma={gamma_rnn:.2f}, th={threshold:.2f}")
        print(f"{'='*60}")

        best_val_acc = 0
        best_model_state = None

        for epoch in range(pretrain_epochs):
            alpha = min(epoch / (args.alpha_anneal_epoch - 1), 1.0)

            # Train
            train_loss = self.train_epoch(net, optimizer, criterion, dataloader, args)

            # Process network
            net = self.process_network(net, args, alpha)

            # Evaluate
            net.rnn.reset_stats() if hasattr(net.rnn, 'reset_stats') else None
            val_stat = self.evaluate(net, criterion, dataloader, args, 'val')

            # Learning rate schedule
            if epoch >= args.alpha_anneal_epoch:
                lr_scheduler.step(val_stat['loss'])

            # Save best model
            if val_stat['accuracy'] > best_val_acc:
                best_val_acc = val_stat['accuracy']
                best_model_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{pretrain_epochs}: Loss={train_loss:.4f}, Val Acc={val_stat['accuracy']*100:.2f}%")

        # Load best model and get final metrics
        net.load_state_dict({k: v.to(args.device) for k, v in best_model_state.items()})
        net.rnn.reset_stats() if hasattr(net.rnn, 'reset_stats') else None
        test_stat_lstm = self.evaluate(net, criterion, dataloader, args, 'test')

        result['lstm_test_acc'] = test_stat_lstm['accuracy']
        result['lstm_test_f1'] = test_stat_lstm['f1_macro']
        result['lstm_test_auc'] = test_stat_lstm['auc_macro']
        result['lstm_weight_sparsity'] = self.get_weight_sparsity(net)

        # Save LSTM model for retraining
        lstm_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        # ==================
        # Phase 2: Retrain DeltaLSTM
        # ==================
        args.hid_type = 'DeltaLSTM'
        args.phase = 'retrain'

        # Create new DeltaLSTM model
        net_delta = net_model.Model(args=args, input_size=n_features, n_classes=n_classes)
        net_delta = net_delta.to(args.device)

        # Load pretrained weights
        net_delta.load_state_dict({k: v.to(args.device) for k, v in lstm_state.items()})

        # Optimizer for retraining
        optimizer = optim.AdamW(net_delta.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)

        print(f"\nRetrain DeltaLSTM: gamma={gamma_rnn:.2f}, th={threshold:.2f}")

        best_val_acc = 0
        best_model_state = None

        for epoch in range(retrain_epochs):
            # Train
            train_loss = self.train_epoch(net_delta, optimizer, criterion, dataloader, args)

            # Process network (alpha=1 for retrain)
            net_delta = self.process_network(net_delta, args, alpha=1.0)

            # Evaluate
            net_delta.rnn.reset_stats()
            val_stat = self.evaluate(net_delta, criterion, dataloader, args, 'val')

            if val_stat['accuracy'] > best_val_acc:
                best_val_acc = val_stat['accuracy']
                best_model_state = {k: v.cpu().clone() for k, v in net_delta.state_dict().items()}

            print(f"  Epoch {epoch+1}/{retrain_epochs}: Loss={train_loss:.4f}, Val Acc={val_stat['accuracy']*100:.2f}%")

        # Load best model and get final metrics
        net_delta.load_state_dict({k: v.to(args.device) for k, v in best_model_state.items()})
        net_delta.rnn.reset_stats()
        test_stat_delta = self.evaluate(net_delta, criterion, dataloader, args, 'test')

        result['delta_test_acc'] = test_stat_delta['accuracy']
        result['delta_test_f1'] = test_stat_delta['f1_macro']
        result['delta_test_auc'] = test_stat_delta['auc_macro']
        result['delta_weight_sparsity'] = self.get_weight_sparsity(net_delta)
        result['delta_temporal_sparsity'] = test_stat_delta.get('sp_delta', 0)
        result['delta_sp_dx'] = test_stat_delta.get('sp_dx', 0)
        result['delta_sp_dh'] = test_stat_delta.get('sp_dh', 0)

        # Count non-zero parameters
        n_nonzero = 0
        for name, param in net_delta.named_parameters():
            n_nonzero += torch.count_nonzero(param.data).item()
        result['n_params_nonzero'] = n_nonzero
        result['param_sparsity'] = 1 - (n_nonzero / n_params)

        # FLOPs
        dense_flops = self.count_flops(net_delta)
        sparse_flops = self.count_sparse_flops(
            net_delta,
            result['delta_temporal_sparsity'],
            result['delta_weight_sparsity']
        )
        result['dense_flops'] = dense_flops
        result['sparse_flops'] = sparse_flops
        result['flops_reduction'] = 1 - (sparse_flops / dense_flops) if dense_flops > 0 else 0

        # Accuracy drop
        result['acc_drop'] = result['lstm_test_acc'] - result['delta_test_acc']
        result['acc_drop_pct'] = (result['acc_drop'] / result['lstm_test_acc']) * 100

        print(f"\n  Results: LSTM Acc={result['lstm_test_acc']*100:.2f}%, "
              f"Delta Acc={result['delta_test_acc']*100:.2f}%, "
              f"W.Sparsity={result['delta_weight_sparsity']*100:.1f}%, "
              f"T.Sparsity={result['delta_temporal_sparsity']*100:.1f}%")

        # Clean up
        del net, net_delta, dataloader
        torch.cuda.empty_cache()

        return result

    def run_study(self, gamma_values, threshold_values, pretrain_epochs=50, retrain_epochs=5):
        """Run full hyperparameter study."""
        self.setup_device()

        total_experiments = len(gamma_values) * len(threshold_values)
        print(f"\n{'='*70}")
        print(f"Starting Hyperparameter Study: {total_experiments} experiments")
        print(f"Gamma values: {gamma_values}")
        print(f"Threshold values: {threshold_values}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")

        exp_num = 0
        for gamma in gamma_values:
            for threshold in threshold_values:
                exp_num += 1
                print(f"\n[Experiment {exp_num}/{total_experiments}]")

                try:
                    result = self.run_single_experiment(
                        gamma_rnn=gamma,
                        threshold=threshold,
                        pretrain_epochs=pretrain_epochs,
                        retrain_epochs=retrain_epochs
                    )
                    self.results.append(result)

                    # Save intermediate results
                    self.save_results()

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

        # Generate final outputs
        self.save_results()
        self.generate_plots()
        self.generate_excel_report()

        print(f"\n{'='*70}")
        print(f"Study Complete! Results saved to: {self.output_dir}")
        print(f"{'='*70}")

    def save_results(self):
        """Save results to JSON."""
        with open(f"{self.output_dir}/results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

    def generate_excel_report(self):
        """Generate Excel report with multiple sheets."""
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
            from openpyxl.utils.dataframe import dataframe_to_rows
        except ImportError:
            print("openpyxl not installed. Installing...")
            os.system("pip install openpyxl")
            from openpyxl import Workbook
            from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
            from openpyxl.utils.dataframe import dataframe_to_rows

        df = pd.DataFrame(self.results)
        excel_path = f"{self.output_dir}/hyperparam_study_results.xlsx"

        # Create workbook
        wb = Workbook()

        # Styles
        header_font = Font(bold=True, size=11)
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font_white = Font(bold=True, size=11, color="FFFFFF")
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        center_align = Alignment(horizontal='center', vertical='center')

        def style_sheet(ws, df_sheet):
            """Apply styling to worksheet."""
            # Style header
            for cell in ws[1]:
                cell.font = header_font_white
                cell.fill = header_fill
                cell.alignment = center_align
                cell.border = thin_border

            # Style data cells
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
                for cell in row:
                    cell.border = thin_border
                    cell.alignment = center_align
                    # Format numbers
                    if isinstance(cell.value, float):
                        if 'acc' in str(ws.cell(1, cell.column).value).lower() or \
                           'f1' in str(ws.cell(1, cell.column).value).lower() or \
                           'auc' in str(ws.cell(1, cell.column).value).lower() or \
                           'sparsity' in str(ws.cell(1, cell.column).value).lower():
                            cell.number_format = '0.00%'
                        else:
                            cell.number_format = '0.00'

            # Auto-adjust column widths
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                ws.column_dimensions[column_letter].width = min(max_length + 2, 20)

        # Sheet 1: Summary
        ws = wb.active
        ws.title = "Summary"

        summary_cols = ['gamma_rnn', 'threshold', 'lstm_test_acc', 'delta_test_acc',
                       'acc_drop', 'delta_weight_sparsity', 'delta_temporal_sparsity', 'flops_reduction']
        df_summary = df[summary_cols].copy()
        df_summary.columns = ['Target Sparsity', 'Threshold', 'LSTM Acc', 'DeltaLSTM Acc',
                             'Acc Drop', 'Weight Sparsity', 'Temporal Sparsity', 'FLOPs Reduction']

        for r in dataframe_to_rows(df_summary, index=False, header=True):
            ws.append(r)
        style_sheet(ws, df_summary)

        # Sheet 2: Accuracy Details
        ws2 = wb.create_sheet("Accuracy Details")
        acc_cols = ['gamma_rnn', 'threshold', 'lstm_test_acc', 'lstm_test_f1', 'lstm_test_auc',
                   'delta_test_acc', 'delta_test_f1', 'delta_test_auc', 'acc_drop_pct']
        df_acc = df[acc_cols].copy()
        df_acc.columns = ['Target Sparsity', 'Threshold', 'LSTM Acc', 'LSTM F1', 'LSTM AUC',
                         'Delta Acc', 'Delta F1', 'Delta AUC', 'Acc Drop (%)']

        for r in dataframe_to_rows(df_acc, index=False, header=True):
            ws2.append(r)
        style_sheet(ws2, df_acc)

        # Sheet 3: Sparsity Details
        ws3 = wb.create_sheet("Sparsity Details")
        sp_cols = ['gamma_rnn', 'threshold', 'delta_weight_sparsity', 'delta_temporal_sparsity',
                  'delta_sp_dx', 'delta_sp_dh', 'param_sparsity']
        df_sp = df[sp_cols].copy()
        df_sp.columns = ['Target Sparsity', 'Threshold', 'Weight Sparsity', 'Temporal Sparsity',
                        'Input Sparsity', 'Hidden Sparsity', 'Param Sparsity']

        for r in dataframe_to_rows(df_sp, index=False, header=True):
            ws3.append(r)
        style_sheet(ws3, df_sp)

        # Sheet 4: FLOPs Analysis
        ws4 = wb.create_sheet("FLOPs Analysis")
        flops_cols = ['gamma_rnn', 'threshold', 'n_params_total', 'n_params_nonzero',
                     'dense_flops', 'sparse_flops', 'flops_reduction']
        df_flops = df[flops_cols].copy()
        df_flops.columns = ['Target Sparsity', 'Threshold', 'Total Params', 'Non-zero Params',
                           'Dense FLOPs', 'Sparse FLOPs', 'FLOPs Reduction']

        for r in dataframe_to_rows(df_flops, index=False, header=True):
            ws4.append(r)
        style_sheet(ws4, df_flops)

        # Sheet 5: Pivot Table - Accuracy vs Gamma/Threshold
        ws5 = wb.create_sheet("Accuracy Pivot")
        pivot_acc = df.pivot_table(
            values='delta_test_acc',
            index='gamma_rnn',
            columns='threshold',
            aggfunc='mean'
        )
        # Add row and column labels
        ws5.cell(1, 1, "Target Sparsity \\ Threshold")
        for j, col in enumerate(pivot_acc.columns):
            ws5.cell(1, j + 2, f"th={col:.2f}")
        for i, idx in enumerate(pivot_acc.index):
            ws5.cell(i + 2, 1, f"γ={idx:.2f}")
            for j, col in enumerate(pivot_acc.columns):
                val = pivot_acc.loc[idx, col]
                ws5.cell(i + 2, j + 2, val)
        style_sheet(ws5, pivot_acc)

        # Sheet 6: Pivot Table - FLOPs Reduction
        ws6 = wb.create_sheet("FLOPs Pivot")
        pivot_flops = df.pivot_table(
            values='flops_reduction',
            index='gamma_rnn',
            columns='threshold',
            aggfunc='mean'
        )
        ws6.cell(1, 1, "Target Sparsity \\ Threshold")
        for j, col in enumerate(pivot_flops.columns):
            ws6.cell(1, j + 2, f"th={col:.2f}")
        for i, idx in enumerate(pivot_flops.index):
            ws6.cell(i + 2, 1, f"γ={idx:.2f}")
            for j, col in enumerate(pivot_flops.columns):
                val = pivot_flops.loc[idx, col]
                ws6.cell(i + 2, j + 2, val)
        style_sheet(ws6, pivot_flops)

        wb.save(excel_path)
        print(f"Excel report saved to: {excel_path}")

    def generate_plots(self):
        """Generate beautiful plots."""
        if not self.results:
            print("No results to plot")
            return

        df = pd.DataFrame(self.results)
        plot_dir = f"{self.output_dir}/plots"

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # ==================
        # Plot 1: Accuracy Heatmap
        # ==================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # LSTM Accuracy pivot
        pivot_lstm = df.pivot_table(values='lstm_test_acc', index='gamma_rnn', columns='threshold')
        pivot_delta = df.pivot_table(values='delta_test_acc', index='gamma_rnn', columns='threshold')

        sns.heatmap(pivot_lstm * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=axes[0], cbar_kws={'label': 'Accuracy (%)'}, vmin=80, vmax=100)
        axes[0].set_title('LSTM Test Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Delta Threshold', fontsize=10)
        axes[0].set_ylabel('Target Sparsity (γ)', fontsize=10)

        sns.heatmap(pivot_delta * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=axes[1], cbar_kws={'label': 'Accuracy (%)'}, vmin=80, vmax=100)
        axes[1].set_title('DeltaLSTM Test Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Delta Threshold', fontsize=10)
        axes[1].set_ylabel('Target Sparsity (γ)', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{plot_dir}/accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{plot_dir}/accuracy_heatmap.pdf', bbox_inches='tight')
        plt.close()

        # ==================
        # Plot 2: Sparsity Heatmaps
        # ==================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        pivot_weight_sp = df.pivot_table(values='delta_weight_sparsity', index='gamma_rnn', columns='threshold')
        pivot_temp_sp = df.pivot_table(values='delta_temporal_sparsity', index='gamma_rnn', columns='threshold')

        sns.heatmap(pivot_weight_sp * 100, annot=True, fmt='.1f', cmap='Blues',
                   ax=axes[0], cbar_kws={'label': 'Sparsity (%)'})
        axes[0].set_title('Weight Sparsity (%)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Delta Threshold', fontsize=10)
        axes[0].set_ylabel('Target Sparsity (γ)', fontsize=10)

        sns.heatmap(pivot_temp_sp * 100, annot=True, fmt='.1f', cmap='Oranges',
                   ax=axes[1], cbar_kws={'label': 'Sparsity (%)'})
        axes[1].set_title('Temporal Sparsity (%)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Delta Threshold', fontsize=10)
        axes[1].set_ylabel('Target Sparsity (γ)', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{plot_dir}/sparsity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{plot_dir}/sparsity_heatmap.pdf', bbox_inches='tight')
        plt.close()

        # ==================
        # Plot 3: FLOPs Reduction Heatmap
        # ==================
        fig, ax = plt.subplots(figsize=(8, 6))

        pivot_flops = df.pivot_table(values='flops_reduction', index='gamma_rnn', columns='threshold')

        sns.heatmap(pivot_flops * 100, annot=True, fmt='.1f', cmap='Greens',
                   ax=ax, cbar_kws={'label': 'FLOPs Reduction (%)'})
        ax.set_title('FLOPs Reduction (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Delta Threshold', fontsize=12)
        ax.set_ylabel('Target Sparsity (γ)', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{plot_dir}/flops_reduction_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{plot_dir}/flops_reduction_heatmap.pdf', bbox_inches='tight')
        plt.close()

        # ==================
        # Plot 4: Accuracy vs Sparsity Trade-off
        # ==================
        fig, ax = plt.subplots(figsize=(10, 6))

        # Combined sparsity metric
        df['combined_sparsity'] = df['delta_weight_sparsity'] * 0.5 + df['delta_temporal_sparsity'] * 0.5

        scatter = ax.scatter(df['flops_reduction'] * 100, df['delta_test_acc'] * 100,
                           c=df['threshold'], cmap='viridis', s=100, edgecolors='black', alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Delta Threshold', fontsize=10)

        # Add Pareto frontier
        pareto_idx = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if (other['flops_reduction'] >= row['flops_reduction'] and
                    other['delta_test_acc'] > row['delta_test_acc']) or \
                   (other['flops_reduction'] > row['flops_reduction'] and
                    other['delta_test_acc'] >= row['delta_test_acc']):
                    dominated = True
                    break
            if not dominated:
                pareto_idx.append(i)

        if pareto_idx:
            pareto_df = df.loc[pareto_idx].sort_values('flops_reduction')
            ax.plot(pareto_df['flops_reduction'] * 100, pareto_df['delta_test_acc'] * 100,
                   'r--', linewidth=2, label='Pareto Frontier', zorder=5)

        ax.set_xlabel('FLOPs Reduction (%)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy vs FLOPs Reduction Trade-off', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{plot_dir}/accuracy_vs_flops.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{plot_dir}/accuracy_vs_flops.pdf', bbox_inches='tight')
        plt.close()

        # ==================
        # Plot 5: Line plots for different thresholds
        # ==================
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        thresholds = sorted(df['threshold'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))

        for th, color in zip(thresholds, colors):
            subset = df[df['threshold'] == th].sort_values('gamma_rnn')
            axes[0].plot(subset['gamma_rnn'] * 100, subset['delta_test_acc'] * 100,
                        'o-', color=color, label=f'th={th:.2f}', linewidth=2, markersize=6)
            axes[1].plot(subset['gamma_rnn'] * 100, subset['delta_weight_sparsity'] * 100,
                        'o-', color=color, label=f'th={th:.2f}', linewidth=2, markersize=6)
            axes[2].plot(subset['gamma_rnn'] * 100, subset['flops_reduction'] * 100,
                        'o-', color=color, label=f'th={th:.2f}', linewidth=2, markersize=6)

        axes[0].set_xlabel('Target Sparsity (%)', fontsize=11)
        axes[0].set_ylabel('Test Accuracy (%)', fontsize=11)
        axes[0].set_title('Accuracy vs Target Sparsity', fontsize=12, fontweight='bold')
        axes[0].legend(loc='lower left', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Target Sparsity (%)', fontsize=11)
        axes[1].set_ylabel('Weight Sparsity (%)', fontsize=11)
        axes[1].set_title('Weight Sparsity vs Target', fontsize=12, fontweight='bold')
        axes[1].legend(loc='lower right', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('Target Sparsity (%)', fontsize=11)
        axes[2].set_ylabel('FLOPs Reduction (%)', fontsize=11)
        axes[2].set_title('FLOPs Reduction vs Target', fontsize=12, fontweight='bold')
        axes[2].legend(loc='lower right', fontsize=8)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{plot_dir}/metrics_vs_target_sparsity.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{plot_dir}/metrics_vs_target_sparsity.pdf', bbox_inches='tight')
        plt.close()

        # ==================
        # Plot 6: Summary Dashboard
        # ==================
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Accuracy heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(pivot_delta * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=ax1, cbar_kws={'label': '%'}, annot_kws={'size': 8})
        ax1.set_title('DeltaLSTM Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Threshold', fontsize=9)
        ax1.set_ylabel('Target Sparsity', fontsize=9)

        # Weight sparsity heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(pivot_weight_sp * 100, annot=True, fmt='.1f', cmap='Blues',
                   ax=ax2, cbar_kws={'label': '%'}, annot_kws={'size': 8})
        ax2.set_title('Weight Sparsity (%)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Threshold', fontsize=9)
        ax2.set_ylabel('Target Sparsity', fontsize=9)

        # Temporal sparsity heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        sns.heatmap(pivot_temp_sp * 100, annot=True, fmt='.1f', cmap='Oranges',
                   ax=ax3, cbar_kws={'label': '%'}, annot_kws={'size': 8})
        ax3.set_title('Temporal Sparsity (%)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Threshold', fontsize=9)
        ax3.set_ylabel('Target Sparsity', fontsize=9)

        # FLOPs reduction heatmap
        ax4 = fig.add_subplot(gs[1, 0])
        sns.heatmap(pivot_flops * 100, annot=True, fmt='.1f', cmap='Greens',
                   ax=ax4, cbar_kws={'label': '%'}, annot_kws={'size': 8})
        ax4.set_title('FLOPs Reduction (%)', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Threshold', fontsize=9)
        ax4.set_ylabel('Target Sparsity', fontsize=9)

        # Accuracy drop heatmap
        ax5 = fig.add_subplot(gs[1, 1])
        pivot_drop = df.pivot_table(values='acc_drop_pct', index='gamma_rnn', columns='threshold')
        sns.heatmap(pivot_drop, annot=True, fmt='.1f', cmap='Reds',
                   ax=ax5, cbar_kws={'label': '%'}, annot_kws={'size': 8})
        ax5.set_title('Accuracy Drop (%)', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Threshold', fontsize=9)
        ax5.set_ylabel('Target Sparsity', fontsize=9)

        # Pareto plot
        ax6 = fig.add_subplot(gs[1, 2])
        scatter = ax6.scatter(df['flops_reduction'] * 100, df['delta_test_acc'] * 100,
                            c=df['gamma_rnn'], cmap='plasma', s=80, edgecolors='black', alpha=0.8)
        if pareto_idx:
            ax6.plot(pareto_df['flops_reduction'] * 100, pareto_df['delta_test_acc'] * 100,
                    'r--', linewidth=2, label='Pareto')
        ax6.set_xlabel('FLOPs Reduction (%)', fontsize=9)
        ax6.set_ylabel('Accuracy (%)', fontsize=9)
        ax6.set_title('Acc vs FLOPs Trade-off', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax6, label='γ')

        # Line plot: Accuracy vs gamma
        ax7 = fig.add_subplot(gs[2, :2])
        for th, color in zip(thresholds, colors):
            subset = df[df['threshold'] == th].sort_values('gamma_rnn')
            ax7.plot(subset['gamma_rnn'] * 100, subset['delta_test_acc'] * 100,
                    'o-', color=color, label=f'th={th:.2f}', linewidth=2, markersize=5)
        ax7.set_xlabel('Target Sparsity (%)', fontsize=10)
        ax7.set_ylabel('Test Accuracy (%)', fontsize=10)
        ax7.set_title('DeltaLSTM Accuracy vs Target Sparsity', fontsize=11, fontweight='bold')
        ax7.legend(loc='lower left', fontsize=8, ncol=3)
        ax7.grid(True, alpha=0.3)

        # Best configurations table
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        # Find best configs
        best_acc = df.loc[df['delta_test_acc'].idxmax()]
        best_efficient = df.loc[(df['delta_test_acc'] > 0.9) & (df['flops_reduction'] == df[df['delta_test_acc'] > 0.9]['flops_reduction'].max())] if len(df[df['delta_test_acc'] > 0.9]) > 0 else df.loc[df['flops_reduction'].idxmax()]

        table_text = (
            f"Best Configurations\n"
            f"{'='*35}\n\n"
            f"Highest Accuracy:\n"
            f"  γ = {best_acc['gamma_rnn']:.2f}, th = {best_acc['threshold']:.2f}\n"
            f"  Acc = {best_acc['delta_test_acc']*100:.2f}%\n"
            f"  FLOPs Red. = {best_acc['flops_reduction']*100:.1f}%\n\n"
        )

        if isinstance(best_efficient, pd.DataFrame) and len(best_efficient) > 0:
            best_efficient = best_efficient.iloc[0]
        if isinstance(best_efficient, pd.Series):
            table_text += (
                f"Best Efficiency (Acc>90%):\n"
                f"  γ = {best_efficient['gamma_rnn']:.2f}, th = {best_efficient['threshold']:.2f}\n"
                f"  Acc = {best_efficient['delta_test_acc']*100:.2f}%\n"
                f"  FLOPs Red. = {best_efficient['flops_reduction']*100:.1f}%"
            )

        ax8.text(0.1, 0.9, table_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        fig.suptitle('DeltaLSTM Hyperparameter Study - GSCDv2', fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(f'{plot_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{plot_dir}/summary_dashboard.pdf', bbox_inches='tight')
        plt.close()

        print(f"Plots saved to: {plot_dir}")


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeltaLSTM Hyperparameter Study')
    parser.add_argument('--dataset_name', default='gscdv2', help='Dataset name')
    parser.add_argument('--gpu_device', default=0, type=int, help='GPU device ID')
    parser.add_argument('--pretrain_epochs', default=50, type=int, help='Pretrain epochs')
    parser.add_argument('--retrain_epochs', default=5, type=int, help='Retrain epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test with reduced params')

    cmd_args = parser.parse_args()

    # Load GSCDv2 arguments
    module_arguments = importlib.import_module(f'modules.{cmd_args.dataset_name}.arguments')
    args_parser = argparse.ArgumentParser()
    args_parser = module_arguments.add_args(args_parser)
    args = args_parser.parse_args([])

    # Override with command line args
    args.dataset_name = cmd_args.dataset_name
    args.gpu_device = cmd_args.gpu_device
    args.use_cuda = 1

    # Set feature paths
    feat_dir = f'./feat/{args.dataset_name}'
    args.trainfile = f'{feat_dir}/TRAIN_D_{args.dataset_name}_NF_40_SI_0.025_ST_0.01_GC_0_FG_1.h5'
    args.valfile = f'{feat_dir}/VAL_D_{args.dataset_name}_NF_40_SI_0.025_ST_0.01_GC_0_FG_1.h5'
    args.testfile = f'{feat_dir}/TEST_D_{args.dataset_name}_NF_40_SI_0.025_ST_0.01_GC_0_FG_1.h5'

    # Define hyperparameter ranges
    if cmd_args.quick:
        # Quick test with fewer values
        gamma_values = [0.0, 0.5, 0.9]
        threshold_values = [0.0, 0.2, 0.4]
        pretrain_epochs = 5
        retrain_epochs = 2
    else:
        # Full study
        gamma_values = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        threshold_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        pretrain_epochs = cmd_args.pretrain_epochs
        retrain_epochs = cmd_args.retrain_epochs

    # Create study
    study = HyperparamStudy(args)

    # Run study
    study.run_study(
        gamma_values=gamma_values,
        threshold_values=threshold_values,
        pretrain_epochs=pretrain_epochs,
        retrain_epochs=retrain_epochs
    )


if __name__ == '__main__':
    main()
