"""
Excel Logger for DeltaLSTM-CBTD
Creates beautifully formatted Excel files with multiple sheets for comprehensive logging.
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

try:
    from openpyxl import Workbook
    from openpyxl.styles import (
        Font, Alignment, Border, Side, PatternFill, NamedStyle
    )
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import LineChart, Reference
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("Warning: openpyxl not available. Install with: pip install openpyxl")


class ExcelLogger:
    """
    Comprehensive Excel logger for training runs.
    Creates multiple sheets with formatted data.
    """

    # Color scheme
    COLORS = {
        'header_bg': '4472C4',      # Blue
        'header_fg': 'FFFFFF',      # White
        'subheader_bg': '8EA9DB',   # Light blue
        'success_bg': 'C6EFCE',     # Light green
        'warning_bg': 'FFEB9C',     # Light yellow
        'error_bg': 'FFC7CE',       # Light red
        'accent1': '5B9BD5',        # Accent blue
        'accent2': 'ED7D31',        # Accent orange
        'accent3': 'A5A5A5',        # Gray
        'delta_color': 'F4B084',    # Orange for DeltaLSTM
    }

    def __init__(self, log_dir: str, model_id: str, dataset_name: str, phase: str):
        """
        Initialize the Excel logger.

        Args:
            log_dir: Directory to save the Excel file
            model_id: Model identifier
            dataset_name: Dataset name (timit, gscdv2)
            phase: Training phase (pretrain, retrain)
        """
        if not OPENPYXL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel logging")

        self.log_dir = log_dir
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.phase = phase

        # Create directory if needed
        os.makedirs(log_dir, exist_ok=True)

        # Excel file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.excel_path = os.path.join(
            log_dir, f"{dataset_name}_{phase}_{timestamp}.xlsx"
        )

        # Initialize workbook
        self.wb = Workbook()
        self.wb.remove(self.wb.active)  # Remove default sheet

        # Data storage
        self.epoch_history: List[Dict] = []
        self.model_config: Dict = {}
        self.model_stats: Dict = {}
        self.layer_breakdown: List[Dict] = []
        self.sparsity_history: List[Dict] = []

        # Create styles
        self._create_styles()

    def _create_styles(self):
        """Create named styles for consistent formatting."""
        # Header style
        self.header_style = NamedStyle(name='header_style')
        self.header_style.font = Font(bold=True, color=self.COLORS['header_fg'], size=11)
        self.header_style.fill = PatternFill(
            start_color=self.COLORS['header_bg'],
            end_color=self.COLORS['header_bg'],
            fill_type='solid'
        )
        self.header_style.alignment = Alignment(horizontal='center', vertical='center')
        self.header_style.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Data style
        self.data_style = NamedStyle(name='data_style')
        self.data_style.alignment = Alignment(horizontal='center', vertical='center')
        self.data_style.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Title style
        self.title_font = Font(bold=True, size=14, color=self.COLORS['header_bg'])
        self.subtitle_font = Font(bold=True, size=12, color=self.COLORS['accent1'])

    def _apply_header_style(self, ws, row: int, start_col: int = 1, end_col: int = None):
        """Apply header styling to a row."""
        if end_col is None:
            end_col = ws.max_column
        for col in range(start_col, end_col + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = Font(bold=True, color=self.COLORS['header_fg'], size=11)
            cell.fill = PatternFill(
                start_color=self.COLORS['header_bg'],
                end_color=self.COLORS['header_bg'],
                fill_type='solid'
            )
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

    def _apply_data_style(self, ws, row: int, start_col: int = 1, end_col: int = None):
        """Apply data styling to a row."""
        if end_col is None:
            end_col = ws.max_column
        for col in range(start_col, end_col + 1):
            cell = ws.cell(row=row, column=col)
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

    def _auto_column_width(self, ws):
        """Auto-adjust column widths."""
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            ws.column_dimensions[column_letter].width = min(max_length + 3, 25)

    def _format_number(self, num) -> str:
        """Format large numbers with K, M, G suffixes."""
        if num is None:
            return '-'
        if isinstance(num, str):
            return num
        if num >= 1e9:
            return f"{num/1e9:.2f}G"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        elif isinstance(num, float):
            if abs(num) < 0.01 and num != 0:
                return f"{num:.2e}"
            return f"{num:.4f}"
        return str(num)

    def set_model_config(self, args, n_features: int, n_classes: int):
        """Store model configuration."""
        self.model_config = {
            'Dataset': args.dataset_name,
            'Phase': args.phase,
            'Model Type': args.hid_type,
            'Hidden Layers': args.hid_layers,
            'Hidden Size': args.hid_size,
            'Input Features': n_features,
            'Output Classes': n_classes,
            'FC Extra Size': args.fc_extra_size,
            'Batch Size': args.batch_size,
            'Learning Rate': args.lr,
            'Epochs': args.n_epochs,
            'Optimizer': getattr(args, 'opt', 'ADAMW'),
            'Weight Decay': getattr(args, 'weight_decay', 0),
            'Dropout': args.hid_dropout,
            'Quantize Activations': args.qa,
            'Activation Bits (int/frac)': f"{args.aqi}/{args.aqf}",
            'Quantize Weights': args.qw,
            'Weight Bits (int/frac)': f"{args.wqi}/{args.wqf}",
            'Seed': args.seed,
        }

        # DeltaLSTM specific config
        if 'Delta' in args.hid_type:
            self.model_config.update({
                'Input Threshold (th_x)': args.th_x,
                'Hidden Threshold (th_h)': args.th_h,
                'Target Sparsity RNN (γ)': f"{args.gamma_rnn*100:.1f}%",
                'Target Sparsity FC (γ)': f"{args.gamma_fc*100:.1f}%",
                'Balance PE': 'Enabled' if args.balance_pe else 'Disabled',
                'Num Arrays': args.num_array,
                'PEs per Array': args.num_array_pe,
            })

    def set_model_stats(self, stats: Dict):
        """Store model statistics."""
        self.model_stats = stats
        if 'layer_breakdown' in stats:
            self.layer_breakdown = stats['layer_breakdown']

    def add_epoch(self, epoch_data: Dict):
        """Add epoch data to history, flattening nested dicts."""
        # Flatten nested dictionaries (like per_class_acc)
        flat_data = {}
        for key, value in epoch_data.items():
            if isinstance(value, dict):
                # Flatten nested dict: {'key': {'a': 1, 'b': 2}} -> {'key_a': 1, 'key_b': 2}
                for sub_key, sub_value in value.items():
                    flat_key = f"{key}_{sub_key}"
                    # Convert numpy types to Python types
                    if hasattr(sub_value, 'item'):
                        sub_value = sub_value.item()
                    flat_data[flat_key] = sub_value
            else:
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    value = value.item()
                flat_data[key] = value
        self.epoch_history.append(flat_data)

    def add_sparsity_snapshot(self, epoch: int, weight_sparsity: Dict, temporal_sparsity: Dict = None):
        """Add sparsity snapshot."""
        snapshot = {'epoch': epoch, **weight_sparsity}
        if temporal_sparsity:
            snapshot.update(temporal_sparsity)
        self.sparsity_history.append(snapshot)

    def _create_summary_sheet(self):
        """Create summary sheet with overview."""
        ws = self.wb.create_sheet("Summary", 0)

        # Title
        ws['A1'] = f"Training Summary - {self.dataset_name.upper()}"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:D1')

        ws['A2'] = f"Phase: {self.phase} | Model: {self.model_config.get('Model Type', 'N/A')}"
        ws['A2'].font = self.subtitle_font
        ws.merge_cells('A2:D2')

        ws['A3'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ws['A3'].font = Font(italic=True, color=self.COLORS['accent3'])

        # Model Config Section
        row = 5
        ws.cell(row=row, column=1, value="Model Configuration")
        ws.cell(row=row, column=1).font = self.subtitle_font
        row += 1

        for key, value in self.model_config.items():
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=str(value))
            ws.cell(row=row, column=1).font = Font(bold=True)
            row += 1

        # Model Stats Section
        if self.model_stats:
            row += 1
            ws.cell(row=row, column=1, value="Model Complexity")
            ws.cell(row=row, column=1).font = self.subtitle_font
            row += 1

            stats_display = [
                ('Total Parameters', self.model_stats.get('total', 0)),
                ('Trainable Parameters', self.model_stats.get('trainable', 0)),
                ('Total FLOPs', self._format_number(self.model_stats.get('flops', 0))),
                ('Total MADDs', self._format_number(self.model_stats.get('madds', 0))),
                ('Parameter Memory', f"{self.model_stats.get('param_memory_mb', 0):.2f} MB"),
                ('Activation Memory', f"{self.model_stats.get('activation_memory_mb', 0):.2f} MB"),
                ('Total Memory', f"{self.model_stats.get('total_memory_mb', 0):.2f} MB"),
            ]

            for label, value in stats_display:
                ws.cell(row=row, column=1, value=label)
                ws.cell(row=row, column=2, value=str(value))
                ws.cell(row=row, column=1).font = Font(bold=True)
                row += 1

        # Best Results Section
        if self.epoch_history:
            row += 1
            ws.cell(row=row, column=1, value="Best Results")
            ws.cell(row=row, column=1).font = self.subtitle_font
            row += 1

            # Find best epoch based on validation metric
            best_epoch = None
            best_val = float('inf')
            metric_key = None

            for ep in self.epoch_history:
                for key in ['VAL_LOSS', 'VAL_PER', 'VAL_ERROR_RATE']:
                    if key in ep:
                        metric_key = key
                        if ep[key] < best_val:
                            best_val = ep[key]
                            best_epoch = ep
                        break

            if best_epoch:
                ws.cell(row=row, column=1, value="Best Epoch")
                ws.cell(row=row, column=2, value=str(best_epoch.get('EPOCH', 0) + 1))
                row += 1

                for key in ['VAL_LOSS', 'VAL_ACCURACY', 'VAL_PER', 'VAL_F1_MACRO',
                           'TEST_LOSS', 'TEST_ACCURACY', 'TEST_PER', 'TEST_F1_MACRO']:
                    if key in best_epoch:
                        label = key.replace('_', ' ').title()
                        value = best_epoch[key]
                        if 'ACCURACY' in key or 'F1' in key:
                            ws.cell(row=row, column=2, value=f"{value*100:.2f}%")
                        elif 'PER' in key or 'ERROR' in key:
                            ws.cell(row=row, column=2, value=f"{value*100:.2f}%")
                        else:
                            ws.cell(row=row, column=2, value=f"{value:.4f}")
                        ws.cell(row=row, column=1, value=label)
                        ws.cell(row=row, column=1).font = Font(bold=True)
                        row += 1

        self._auto_column_width(ws)

    def _create_config_sheet(self):
        """Create model configuration sheet."""
        ws = self.wb.create_sheet("Configuration")

        # Title
        ws['A1'] = "Model Configuration"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:C1')

        # Headers
        ws['A3'] = "Parameter"
        ws['B3'] = "Value"
        ws['C3'] = "Description"
        self._apply_header_style(ws, 3, 1, 3)

        # Config data
        descriptions = {
            'Dataset': 'Training dataset name',
            'Phase': 'Training phase (pretrain/retrain)',
            'Model Type': 'RNN architecture type',
            'Hidden Layers': 'Number of RNN layers',
            'Hidden Size': 'Hidden state dimension',
            'Input Features': 'Input feature dimension',
            'Output Classes': 'Number of output classes',
            'Batch Size': 'Training batch size',
            'Learning Rate': 'Initial learning rate',
            'Epochs': 'Total training epochs',
            'Input Threshold (th_x)': 'Delta threshold for input',
            'Hidden Threshold (th_h)': 'Delta threshold for hidden state',
            'Target Sparsity RNN (γ)': 'Target weight sparsity for RNN',
            'Balance PE': 'Column-balanced PE workload',
        }

        row = 4
        for key, value in self.model_config.items():
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=str(value))
            ws.cell(row=row, column=3, value=descriptions.get(key, ''))
            self._apply_data_style(ws, row, 1, 3)
            row += 1

        self._auto_column_width(ws)

    def _create_complexity_sheet(self):
        """Create model complexity sheet."""
        ws = self.wb.create_sheet("Complexity")

        # Title
        ws['A1'] = "Model Complexity Analysis"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:C1')

        # Summary stats
        ws['A3'] = "Metric"
        ws['B3'] = "Value"
        ws['C3'] = "Unit"
        self._apply_header_style(ws, 3, 1, 3)

        # Support both old and new key names from model_stats
        n_params_total = self.model_stats.get('n_params_total', self.model_stats.get('total', 0))
        n_params_trainable = self.model_stats.get('n_params_trainable', self.model_stats.get('trainable', 0))
        n_params_non_trainable = n_params_total - n_params_trainable
        memory_params = self.model_stats.get('memory_params_mb', self.model_stats.get('param_memory_mb', 0))
        memory_activations = self.model_stats.get('memory_activations_mb', self.model_stats.get('activation_memory_mb', 0))
        memory_total = self.model_stats.get('memory_total_mb', self.model_stats.get('total_memory_mb', 0))

        stats_data = [
            ('Total Parameters', f"{n_params_total:,}", 'params'),
            ('Trainable Parameters', f"{n_params_trainable:,}", 'params'),
            ('Non-trainable Parameters', f"{n_params_non_trainable:,}", 'params'),
            ('Total FLOPs', self._format_number(self.model_stats.get('flops', 0)), 'FLOPs'),
            ('Total MADDs', self._format_number(self.model_stats.get('madds', 0)), 'MADDs'),
            ('Parameter Memory', f"{memory_params:.2f}", 'MB'),
            ('Activation Memory', f"{memory_activations:.2f}", 'MB'),
            ('Total Memory', f"{memory_total:.2f}", 'MB'),
            ('Sequence Length', str(self.model_stats.get('seq_len', 0)), 'frames'),
            ('Input Features', str(self.model_stats.get('n_features', 0)), 'features'),
        ]

        row = 4
        for metric, value, unit in stats_data:
            ws.cell(row=row, column=1, value=metric)
            ws.cell(row=row, column=2, value=value)
            ws.cell(row=row, column=3, value=unit)
            self._apply_data_style(ws, row, 1, 3)
            row += 1

        # Layer breakdown
        if self.layer_breakdown:
            row += 2
            ws.cell(row=row, column=1, value="Layer Breakdown")
            ws.cell(row=row, column=1).font = self.subtitle_font
            row += 1

            has_flops = any(l.get('flops', 0) > 0 for l in self.layer_breakdown)
            n_cols = 6 if has_flops else 4
            ws.cell(row=row, column=1, value="Layer")
            ws.cell(row=row, column=2, value="Type")
            ws.cell(row=row, column=3, value="Parameters")
            ws.cell(row=row, column=4, value="% Params")
            if has_flops:
                ws.cell(row=row, column=5, value="FLOPs")
                ws.cell(row=row, column=6, value="% FLOPs")
            self._apply_header_style(ws, row, 1, n_cols)
            row += 1

            total_params = sum(l['params'] for l in self.layer_breakdown)
            total_flops = sum(l.get('flops', 0) for l in self.layer_breakdown)
            for layer in self.layer_breakdown:
                pct_p = (layer['params'] / total_params * 100) if total_params > 0 else 0
                ws.cell(row=row, column=1, value=layer['name'])
                ws.cell(row=row, column=2, value=layer['type'])
                ws.cell(row=row, column=3, value=f"{layer['params']:,}")
                ws.cell(row=row, column=4, value=f"{pct_p:.1f}%")
                if has_flops:
                    lf = layer.get('flops', 0)
                    pct_f = (lf / total_flops * 100) if total_flops > 0 else 0
                    ws.cell(row=row, column=5, value=f"{lf:,}")
                    ws.cell(row=row, column=6, value=f"{pct_f:.1f}%")
                self._apply_data_style(ws, row, 1, n_cols)
                row += 1

        self._auto_column_width(ws)

    def _create_training_history_sheet(self):
        """Create training history sheet."""
        if not self.epoch_history:
            return

        ws = self.wb.create_sheet("Training History")

        # Title
        ws['A1'] = "Training History"
        ws['A1'].font = self.title_font

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(self.epoch_history)

        # Reorder columns for better readability
        priority_cols = ['EPOCH', 'TIME_CURR', 'LR_CURR', 'ALPHA',
                        'TRAIN_LOSS', 'TRAIN_REG',
                        'VAL_LOSS', 'VAL_ACCURACY', 'VAL_PER', 'VAL_F1_MACRO',
                        'TEST_LOSS', 'TEST_ACCURACY', 'TEST_PER', 'TEST_F1_MACRO',
                        'SP_W_RNN', 'SP_W_FC']

        ordered_cols = [c for c in priority_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + other_cols]

        # Write headers
        for col_idx, col_name in enumerate(df.columns, 1):
            ws.cell(row=3, column=col_idx, value=col_name)
        self._apply_header_style(ws, 3, 1, len(df.columns))

        # Write data
        for row_idx, row_data in enumerate(df.values, 4):
            for col_idx, value in enumerate(row_data, 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                # Handle numpy types
                if hasattr(value, 'item'):
                    value = value.item()
                # Handle different value types
                if isinstance(value, float):
                    if abs(value) < 1:
                        cell.value = f"{value:.4f}"
                    else:
                        cell.value = f"{value:.2f}"
                elif isinstance(value, dict):
                    # Convert dict to string if somehow passed through
                    cell.value = str(value)
                else:
                    cell.value = value
            self._apply_data_style(ws, row_idx, 1, len(df.columns))

        self._auto_column_width(ws)

        # Add chart if possible
        if len(df) > 1 and 'TRAIN_LOSS' in df.columns:
            self._add_loss_chart(ws, len(df))

    def _add_loss_chart(self, ws, num_rows: int):
        """Add a loss chart to the training history sheet."""
        try:
            chart = LineChart()
            chart.title = "Training Progress"
            chart.x_axis.title = "Epoch"
            chart.y_axis.title = "Loss"
            chart.style = 10

            # Find column indices
            df = pd.DataFrame(self.epoch_history)
            cols = list(df.columns)

            train_col = cols.index('TRAIN_LOSS') + 1 if 'TRAIN_LOSS' in cols else None
            val_col = cols.index('VAL_LOSS') + 1 if 'VAL_LOSS' in cols else None

            if train_col:
                data = Reference(ws, min_col=train_col, min_row=3, max_row=3 + num_rows)
                chart.add_data(data, titles_from_data=True)

            if val_col:
                data = Reference(ws, min_col=val_col, min_row=3, max_row=3 + num_rows)
                chart.add_data(data, titles_from_data=True)

            # Position chart
            chart.width = 15
            chart.height = 10
            ws.add_chart(chart, f"{get_column_letter(len(cols) + 2)}3")
        except Exception as e:
            pass  # Chart creation is optional

    def _create_metrics_sheet(self):
        """Create detailed metrics sheet."""
        if not self.epoch_history:
            return

        ws = self.wb.create_sheet("Metrics Detail")

        # Title
        ws['A1'] = "Detailed Metrics"
        ws['A1'].font = self.title_font

        df = pd.DataFrame(self.epoch_history)

        # Accuracy metrics
        acc_cols = [c for c in df.columns if 'ACCURACY' in c or 'F1' in c or 'AUC' in c]
        if acc_cols:
            row = 3
            ws.cell(row=row, column=1, value="Accuracy Metrics")
            ws.cell(row=row, column=1).font = self.subtitle_font
            row += 1

            headers = ['Epoch'] + acc_cols
            for col_idx, header in enumerate(headers, 1):
                ws.cell(row=row, column=col_idx, value=header)
            self._apply_header_style(ws, row, 1, len(headers))
            row += 1

            for _, epoch_data in df.iterrows():
                ws.cell(row=row, column=1, value=epoch_data.get('EPOCH', 0) + 1)
                for col_idx, col in enumerate(acc_cols, 2):
                    val = epoch_data.get(col, 0)
                    ws.cell(row=row, column=col_idx, value=f"{val*100:.2f}%" if val else '-')
                self._apply_data_style(ws, row, 1, len(headers))
                row += 1

        # Error metrics
        err_cols = [c for c in df.columns if 'PER' in c or 'ERROR' in c or 'EER' in c or 'FAR' in c]
        if err_cols:
            row += 2
            ws.cell(row=row, column=1, value="Error Metrics")
            ws.cell(row=row, column=1).font = self.subtitle_font
            row += 1

            headers = ['Epoch'] + err_cols
            for col_idx, header in enumerate(headers, 1):
                ws.cell(row=row, column=col_idx, value=header)
            self._apply_header_style(ws, row, 1, len(headers))
            row += 1

            for _, epoch_data in df.iterrows():
                ws.cell(row=row, column=1, value=epoch_data.get('EPOCH', 0) + 1)
                for col_idx, col in enumerate(err_cols, 2):
                    val = epoch_data.get(col, 0)
                    ws.cell(row=row, column=col_idx, value=f"{val*100:.2f}%" if val else '-')
                self._apply_data_style(ws, row, 1, len(headers))
                row += 1

        self._auto_column_width(ws)

    def _create_sparsity_sheet(self):
        """Create sparsity analysis sheet."""
        ws = self.wb.create_sheet("Sparsity")

        # Title
        ws['A1'] = "Sparsity Analysis"
        ws['A1'].font = self.title_font

        if not self.epoch_history:
            return

        df = pd.DataFrame(self.epoch_history)

        # Weight sparsity over epochs
        sparsity_cols = [c for c in df.columns if 'SP_W' in c or 'SP_D' in c]
        if sparsity_cols:
            row = 3
            ws.cell(row=row, column=1, value="Weight Sparsity History")
            ws.cell(row=row, column=1).font = self.subtitle_font
            row += 1

            headers = ['Epoch'] + sparsity_cols
            for col_idx, header in enumerate(headers, 1):
                ws.cell(row=row, column=col_idx, value=header)
            self._apply_header_style(ws, row, 1, len(headers))
            row += 1

            for _, epoch_data in df.iterrows():
                ws.cell(row=row, column=1, value=epoch_data.get('EPOCH', 0) + 1)
                for col_idx, col in enumerate(sparsity_cols, 2):
                    val = epoch_data.get(col, 0)
                    cell = ws.cell(row=row, column=col_idx)
                    cell.value = f"{val*100:.2f}%" if val else '-'
                    # Color code based on sparsity level
                    if val and val > 0.8:
                        cell.fill = PatternFill(start_color=self.COLORS['success_bg'],
                                               end_color=self.COLORS['success_bg'],
                                               fill_type='solid')
                    elif val and val > 0.5:
                        cell.fill = PatternFill(start_color=self.COLORS['warning_bg'],
                                               end_color=self.COLORS['warning_bg'],
                                               fill_type='solid')
                self._apply_data_style(ws, row, 1, len(headers))
                row += 1

        # Temporal sparsity (DeltaLSTM)
        delta_cols = [c for c in df.columns if 'SP_DX' in c or 'SP_DH' in c or 'SP_DXH' in c]
        if delta_cols:
            row += 2
            ws.cell(row=row, column=1, value="Temporal Sparsity (DeltaLSTM)")
            ws.cell(row=row, column=1).font = self.subtitle_font
            ws.cell(row=row, column=1).fill = PatternFill(
                start_color=self.COLORS['delta_color'],
                end_color=self.COLORS['delta_color'],
                fill_type='solid'
            )
            row += 1

            headers = ['Epoch'] + delta_cols
            for col_idx, header in enumerate(headers, 1):
                ws.cell(row=row, column=col_idx, value=header)
            self._apply_header_style(ws, row, 1, len(headers))
            row += 1

            for _, epoch_data in df.iterrows():
                ws.cell(row=row, column=1, value=epoch_data.get('EPOCH', 0) + 1)
                for col_idx, col in enumerate(delta_cols, 2):
                    val = epoch_data.get(col, 0)
                    ws.cell(row=row, column=col_idx, value=f"{val*100:.2f}%" if val else '-')
                self._apply_data_style(ws, row, 1, len(headers))
                row += 1

        self._auto_column_width(ws)

    def _create_deltalstm_sheet(self):
        """Create DeltaLSTM specific sheet."""
        if 'Delta' not in self.model_config.get('Model Type', ''):
            return

        ws = self.wb.create_sheet("DeltaLSTM")

        # Title with delta color
        ws['A1'] = "DeltaLSTM Configuration & Statistics"
        ws['A1'].font = Font(bold=True, size=14, color=self.COLORS['delta_color'])
        ws.merge_cells('A1:C1')

        # Configuration
        row = 3
        ws.cell(row=row, column=1, value="Configuration")
        ws.cell(row=row, column=1).font = self.subtitle_font
        row += 1

        ws.cell(row=row, column=1, value="Parameter")
        ws.cell(row=row, column=2, value="Value")
        ws.cell(row=row, column=3, value="Description")
        self._apply_header_style(ws, row, 1, 3)
        row += 1

        delta_config = [
            ('Input Threshold (th_x)', self.model_config.get('Input Threshold (th_x)', 0),
             'Minimum delta magnitude for input updates'),
            ('Hidden Threshold (th_h)', self.model_config.get('Hidden Threshold (th_h)', 0),
             'Minimum delta magnitude for hidden state updates'),
            ('Target Sparsity (γ)', self.model_config.get('Target Sparsity RNN (γ)', 0),
             'Target weight sparsity via CBTD'),
            ('Balance PE', self.model_config.get('Balance PE', 'Disabled'),
             'Column-balanced PE workload distribution'),
            ('Num Arrays', self.model_config.get('Num Arrays', 0),
             'Number of MAC arrays'),
            ('PEs per Array', self.model_config.get('PEs per Array', 0),
             'Processing elements per array'),
        ]

        for param, value, desc in delta_config:
            ws.cell(row=row, column=1, value=param)
            ws.cell(row=row, column=2, value=str(value))
            ws.cell(row=row, column=3, value=desc)
            self._apply_data_style(ws, row, 1, 3)
            row += 1

        # Sparsity benefits
        if self.epoch_history:
            df = pd.DataFrame(self.epoch_history)

            row += 2
            ws.cell(row=row, column=1, value="Sparsity Benefits (Final Epoch)")
            ws.cell(row=row, column=1).font = self.subtitle_font
            row += 1

            last_epoch = df.iloc[-1] if len(df) > 0 else {}

            benefits = [
                ('Weight Sparsity (RNN)', last_epoch.get('SP_W_RNN', 0)),
                ('Input Delta Sparsity (Δx)', last_epoch.get('VAL_SP_DX', last_epoch.get('TEST_SP_DX', 0))),
                ('Hidden Delta Sparsity (Δh)', last_epoch.get('VAL_SP_DH', last_epoch.get('TEST_SP_DH', 0))),
                ('Combined Delta Sparsity (Δxh)', last_epoch.get('VAL_SP_DXH', last_epoch.get('TEST_SP_DXH', 0))),
            ]

            ws.cell(row=row, column=1, value="Metric")
            ws.cell(row=row, column=2, value="Value")
            ws.cell(row=row, column=3, value="Compute Savings")
            self._apply_header_style(ws, row, 1, 3)
            row += 1

            for metric, value in benefits:
                ws.cell(row=row, column=1, value=metric)
                ws.cell(row=row, column=2, value=f"{value*100:.2f}%" if value else '-')
                ws.cell(row=row, column=3, value=f"~{value*100:.0f}% fewer ops" if value else '-')
                self._apply_data_style(ws, row, 1, 3)

                # Color code
                if value and value > 0.7:
                    for col in range(1, 4):
                        ws.cell(row=row, column=col).fill = PatternFill(
                            start_color=self.COLORS['success_bg'],
                            end_color=self.COLORS['success_bg'],
                            fill_type='solid'
                        )
                row += 1

        self._auto_column_width(ws)

    def save(self):
        """Save the Excel file with all sheets."""
        # Create all sheets
        self._create_summary_sheet()
        self._create_config_sheet()
        if self.model_stats:
            self._create_complexity_sheet()
        self._create_training_history_sheet()
        self._create_metrics_sheet()
        self._create_sparsity_sheet()
        self._create_deltalstm_sheet()

        # Save workbook
        self.wb.save(self.excel_path)
        return self.excel_path

    def get_path(self) -> str:
        """Get the Excel file path."""
        return self.excel_path


def create_excel_logger(args, model_id: str) -> Optional[ExcelLogger]:
    """
    Factory function to create an ExcelLogger.

    Args:
        args: Arguments namespace with log_dir, dataset_name, phase
        model_id: Model identifier

    Returns:
        ExcelLogger instance or None if openpyxl not available
    """
    if not OPENPYXL_AVAILABLE:
        return None

    log_dir = getattr(args, 'log_dir', './log')
    if hasattr(args, 'dataset_name'):
        log_dir = os.path.join(log_dir, args.dataset_name)

    return ExcelLogger(
        log_dir=log_dir,
        model_id=model_id,
        dataset_name=args.dataset_name,
        phase=args.phase
    )
