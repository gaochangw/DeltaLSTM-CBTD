"""
Training Plotter - Real-time visualization of training metrics.
Generates plots organized by dataset and phase, mirroring the log/save folder structure.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures


class TrainingPlotter:
    """Generates and updates training visualization plots after each epoch."""

    def __init__(self, plot_dir, model_id, dataset_name, phase):
        """
        Initialize the training plotter.

        Args:
            plot_dir: Base directory for plots (e.g., './plot')
            model_id: Model identifier string for filenames
            dataset_name: Name of dataset (e.g., 'gscdv2', 'timit')
            phase: Training phase ('pretrain' or 'retrain')
        """
        self.base_dir = os.path.join(plot_dir, dataset_name, phase)
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.phase = phase

        # Organized subdirectories
        self.dirs = {
            'metrics': os.path.join(self.base_dir, '01_metrics'),
            'training': os.path.join(self.base_dir, '02_training'),
            'sparsity': os.path.join(self.base_dir, '03_sparsity'),
            'summary': os.path.join(self.base_dir, '04_summary'),
        }

        # Create all plot directories
        for subdir in self.dirs.values():
            os.makedirs(subdir, exist_ok=True)

        # For backward compatibility
        self.plot_dir = self.base_dir

        # Store history for plotting
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'val_accuracy': [],
            'test_accuracy': [],
            'val_f1_macro': [],
            'test_f1_macro': [],
            'val_auc_macro': [],
            'test_auc_macro': [],
            'lr': [],
            'alpha': [],
            'sp_w_rnn': [],
            'sp_w_fc': [],
            # DeltaLSTM specific
            'sp_dx': [],
            'sp_dh': [],
            'balance': [],
            # TIMIT specific (PER)
            'val_per': [],
            'test_per': [],
        }

        # Plot styling
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'train': '#2ecc71',
            'val': '#3498db',
            'test': '#e74c3c',
            'lr': '#9b59b6',
            'sparsity': '#f39c12',
        }

    def update(self, log_stat):
        """
        Update history with new epoch data and regenerate plots.

        Args:
            log_stat: Dictionary containing metrics from current epoch
        """
        # Extract and store metrics
        self.history['epoch'].append(log_stat.get('EPOCH', len(self.history['epoch'])))
        self.history['train_loss'].append(log_stat.get('TRAIN_LOSS', None))
        self.history['val_loss'].append(log_stat.get('VAL_LOSS', None))
        self.history['test_loss'].append(log_stat.get('TEST_LOSS', None))
        self.history['val_accuracy'].append(log_stat.get('VAL_ACCURACY', None))
        self.history['test_accuracy'].append(log_stat.get('TEST_ACCURACY', None))
        self.history['val_f1_macro'].append(log_stat.get('VAL_F1_MACRO', None))
        self.history['test_f1_macro'].append(log_stat.get('TEST_F1_MACRO', None))
        self.history['val_auc_macro'].append(log_stat.get('VAL_AUC_MACRO', None))
        self.history['test_auc_macro'].append(log_stat.get('TEST_AUC_MACRO', None))
        self.history['lr'].append(log_stat.get('LR_CURR', None))
        self.history['alpha'].append(log_stat.get('ALPHA', None))
        self.history['sp_w_rnn'].append(log_stat.get('SP_W_RNN', None))
        self.history['sp_w_fc'].append(log_stat.get('SP_W_FC', None))

        # DeltaLSTM specific
        self.history['sp_dx'].append(log_stat.get('VAL_SP_DX', log_stat.get('TEST_SP_DX', None)))
        self.history['sp_dh'].append(log_stat.get('VAL_SP_DH', log_stat.get('TEST_SP_DH', None)))
        self.history['balance'].append(log_stat.get('VAL_BALANCE', log_stat.get('TEST_BALANCE', None)))

        # TIMIT specific (PER - Phoneme Error Rate)
        self.history['val_per'].append(log_stat.get('VAL_PER', None))
        self.history['test_per'].append(log_stat.get('TEST_PER', None))

        # Generate all plots
        self._plot_all()

    def _plot_all(self):
        """Generate all relevant plots based on available data."""
        epochs = self.history['epoch']
        if len(epochs) < 1:
            return

        # 1. Loss curves
        self._plot_loss_curves(epochs)

        # 2. Accuracy/Error curves (dataset-specific)
        if self.dataset_name == 'timit':
            self._plot_per_curves(epochs)
        else:
            self._plot_accuracy_curves(epochs)

        # 3. F1 and AUC curves (for classification)
        if self.dataset_name != 'timit':
            self._plot_f1_curves(epochs)
            self._plot_auc_curves(epochs)

        # 4. Learning rate and alpha
        self._plot_lr_alpha(epochs)

        # 5. Weight sparsity
        self._plot_weight_sparsity(epochs)

        # 6. DeltaLSTM temporal sparsity (if available)
        if any(v is not None for v in self.history['sp_dx']):
            self._plot_temporal_sparsity(epochs)

        # 7. Summary dashboard
        self._plot_dashboard(epochs)

    def _plot_loss_curves(self, epochs):
        """Plot training, validation, and test loss."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if any(v is not None for v in self.history['train_loss']):
            ax.plot(epochs, self.history['train_loss'],
                   label='Train', color=self.colors['train'], linewidth=2)
        if any(v is not None for v in self.history['val_loss']):
            ax.plot(epochs, self.history['val_loss'],
                   label='Validation', color=self.colors['val'], linewidth=2)
        if any(v is not None for v in self.history['test_loss']):
            ax.plot(epochs, self.history['test_loss'],
                   label='Test', color=self.colors['test'], linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Loss Curves - {self.dataset_name.upper()} ({self.phase})', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['metrics'], f'{self.model_id}_loss.png'), dpi=150)
        plt.close(fig)

    def _plot_accuracy_curves(self, epochs):
        """Plot validation and test accuracy."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if any(v is not None for v in self.history['val_accuracy']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['val_accuracy']],
                   label='Validation', color=self.colors['val'], linewidth=2)
        if any(v is not None for v in self.history['test_accuracy']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['test_accuracy']],
                   label='Test', color=self.colors['test'], linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Accuracy - {self.dataset_name.upper()} ({self.phase})', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['metrics'], f'{self.model_id}_accuracy.png'), dpi=150)
        plt.close(fig)

    def _plot_per_curves(self, epochs):
        """Plot Phoneme Error Rate for TIMIT."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if any(v is not None for v in self.history['val_per']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['val_per']],
                   label='Validation', color=self.colors['val'], linewidth=2)
        if any(v is not None for v in self.history['test_per']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['test_per']],
                   label='Test', color=self.colors['test'], linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Phoneme Error Rate (%)', fontsize=12)
        ax.set_title(f'PER - {self.dataset_name.upper()} ({self.phase})', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['metrics'], f'{self.model_id}_per.png'), dpi=150)
        plt.close(fig)

    def _plot_f1_curves(self, epochs):
        """Plot F1 macro scores."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if any(v is not None for v in self.history['val_f1_macro']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['val_f1_macro']],
                   label='Validation', color=self.colors['val'], linewidth=2)
        if any(v is not None for v in self.history['test_f1_macro']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['test_f1_macro']],
                   label='Test', color=self.colors['test'], linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('F1 Score (Macro) (%)', fontsize=12)
        ax.set_title(f'F1 Score - {self.dataset_name.upper()} ({self.phase})', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['metrics'], f'{self.model_id}_f1.png'), dpi=150)
        plt.close(fig)

    def _plot_auc_curves(self, epochs):
        """Plot AUC macro scores."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if any(v is not None for v in self.history['val_auc_macro']):
            ax.plot(epochs, self.history['val_auc_macro'],
                   label='Validation', color=self.colors['val'], linewidth=2)
        if any(v is not None for v in self.history['test_auc_macro']):
            ax.plot(epochs, self.history['test_auc_macro'],
                   label='Test', color=self.colors['test'], linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('AUC (Macro)', fontsize=12)
        ax.set_title(f'AUC-ROC - {self.dataset_name.upper()} ({self.phase})', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['metrics'], f'{self.model_id}_auc.png'), dpi=150)
        plt.close(fig)

    def _plot_lr_alpha(self, epochs):
        """Plot learning rate and alpha annealing."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Learning rate
        if any(v is not None for v in self.history['lr']):
            ax1.plot(epochs, self.history['lr'],
                    color=self.colors['lr'], linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Learning Rate', fontsize=12)
            ax1.set_title('Learning Rate Schedule', fontsize=14)
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)

        # Alpha (CBTD annealing)
        if any(v is not None for v in self.history['alpha']):
            ax2.plot(epochs, self.history['alpha'],
                    color=self.colors['sparsity'], linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Alpha', fontsize=12)
            ax2.set_title('CBTD Alpha Annealing', fontsize=14)
            ax2.set_ylim([0, 1.1])
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['training'], f'{self.model_id}_lr_alpha.png'), dpi=150)
        plt.close(fig)

    def _plot_weight_sparsity(self, epochs):
        """Plot weight sparsity of RNN and FC layers."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if any(v is not None for v in self.history['sp_w_rnn']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['sp_w_rnn']],
                   label='RNN Weights', color='#e74c3c', linewidth=2)
        if any(v is not None and v > 0 for v in self.history['sp_w_fc']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['sp_w_fc']],
                   label='FC Weights', color='#3498db', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Weight Sparsity (%)', fontsize=12)
        ax.set_title(f'Weight Sparsity - {self.dataset_name.upper()} ({self.phase})', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['sparsity'], f'{self.model_id}_weight_sparsity.png'), dpi=150)
        plt.close(fig)

    def _plot_temporal_sparsity(self, epochs):
        """Plot temporal sparsity for DeltaLSTM."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Delta sparsity (individual)
        if any(v is not None for v in self.history['sp_dx']):
            ax1.plot(epochs, [v * 100 if v else None for v in self.history['sp_dx']],
                    label='Delta X', color='#2ecc71', linewidth=2)
        if any(v is not None for v in self.history['sp_dh']):
            ax1.plot(epochs, [v * 100 if v else None for v in self.history['sp_dh']],
                    label='Delta H', color='#e74c3c', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Temporal Sparsity (%)', fontsize=12)
        ax1.set_title('DeltaLSTM Temporal Sparsity (Individual)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])

        # Combined/Averaged temporal sparsity (as if concatenated over all layers)
        sp_avg = []
        for dx, dh in zip(self.history['sp_dx'], self.history['sp_dh']):
            if dx is not None and dh is not None:
                # Average of delta x and delta h (treating as concatenated)
                sp_avg.append((dx + dh) / 2 * 100)
            elif dx is not None:
                sp_avg.append(dx * 100)
            elif dh is not None:
                sp_avg.append(dh * 100)
            else:
                sp_avg.append(None)

        if any(v is not None for v in sp_avg):
            ax2.plot(epochs, sp_avg, color='#3498db', linewidth=2, marker='o', markersize=4)
            ax2.fill_between(epochs, 0, sp_avg, alpha=0.3, color='#3498db')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Avg Temporal Sparsity (%)', fontsize=12)
            ax2.set_title('Combined Temporal Sparsity\n(Avg over Delta X & H)', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 100])
            # Add final value annotation
            final_val = [v for v in sp_avg if v is not None]
            if final_val:
                ax2.annotate(f'{final_val[-1]:.1f}%',
                            xy=(epochs[-1], final_val[-1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold')

        # Workload balance
        if any(v is not None for v in self.history['balance']):
            ax3.plot(epochs, self.history['balance'],
                    color='#9b59b6', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Workload Balance', fontsize=12)
            ax3.set_title('PE Workload Balance', fontsize=14)
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['sparsity'], f'{self.model_id}_temporal_sparsity.png'), dpi=150)
        plt.close(fig)

    def _plot_dashboard(self, epochs):
        """Create a summary dashboard with key metrics."""
        if self.dataset_name == 'timit':
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        fig.suptitle(f'Training Dashboard - {self.dataset_name.upper()} ({self.phase})\n{self.model_id}',
                    fontsize=14, fontweight='bold')

        # Loss
        ax = axes[0, 0]
        if any(v is not None for v in self.history['train_loss']):
            ax.plot(epochs, self.history['train_loss'], label='Train', color=self.colors['train'], linewidth=2)
        if any(v is not None for v in self.history['val_loss']):
            ax.plot(epochs, self.history['val_loss'], label='Val', color=self.colors['val'], linewidth=2)
        if any(v is not None for v in self.history['test_loss']):
            ax.plot(epochs, self.history['test_loss'], label='Test', color=self.colors['test'], linewidth=2)
        ax.set_title('Loss')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        if self.dataset_name == 'timit':
            # PER for TIMIT
            ax = axes[0, 1]
            if any(v is not None for v in self.history['val_per']):
                ax.plot(epochs, [v * 100 if v else None for v in self.history['val_per']],
                       label='Val', color=self.colors['val'], linewidth=2)
            if any(v is not None for v in self.history['test_per']):
                ax.plot(epochs, [v * 100 if v else None for v in self.history['test_per']],
                       label='Test', color=self.colors['test'], linewidth=2)
            ax.set_title('Phoneme Error Rate (%)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            # Accuracy for classification
            ax = axes[0, 1]
            if any(v is not None for v in self.history['val_accuracy']):
                ax.plot(epochs, [v * 100 if v else None for v in self.history['val_accuracy']],
                       label='Val', color=self.colors['val'], linewidth=2)
            if any(v is not None for v in self.history['test_accuracy']):
                ax.plot(epochs, [v * 100 if v else None for v in self.history['test_accuracy']],
                       label='Test', color=self.colors['test'], linewidth=2)
            ax.set_title('Accuracy (%)')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # F1 Score
            ax = axes[0, 2]
            if any(v is not None for v in self.history['val_f1_macro']):
                ax.plot(epochs, [v * 100 if v else None for v in self.history['val_f1_macro']],
                       label='Val', color=self.colors['val'], linewidth=2)
            if any(v is not None for v in self.history['test_f1_macro']):
                ax.plot(epochs, [v * 100 if v else None for v in self.history['test_f1_macro']],
                       label='Test', color=self.colors['test'], linewidth=2)
            ax.set_title('F1 Score (Macro) (%)')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Learning Rate
        ax = axes[1, 0]
        if any(v is not None for v in self.history['lr']):
            ax.plot(epochs, self.history['lr'], color=self.colors['lr'], linewidth=2)
        ax.set_title('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Weight Sparsity
        ax = axes[1, 1]
        if any(v is not None for v in self.history['sp_w_rnn']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['sp_w_rnn']],
                   label='RNN', color='#e74c3c', linewidth=2)
        if any(v is not None and v > 0 for v in self.history['sp_w_fc']):
            ax.plot(epochs, [v * 100 if v else None for v in self.history['sp_w_fc']],
                   label='FC', color='#3498db', linewidth=2)
        ax.set_title('Weight Sparsity (%)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        if self.dataset_name != 'timit':
            # AUC for classification
            ax = axes[1, 2]
            if any(v is not None for v in self.history['val_auc_macro']):
                ax.plot(epochs, self.history['val_auc_macro'],
                       label='Val', color=self.colors['val'], linewidth=2)
            if any(v is not None for v in self.history['test_auc_macro']):
                ax.plot(epochs, self.history['test_auc_macro'],
                       label='Test', color=self.colors['test'], linewidth=2)
            ax.set_title('AUC-ROC (Macro)')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['summary'], f'{self.model_id}_dashboard.png'), dpi=150)
        plt.close(fig)

    def save_history_csv(self):
        """Save the training history to a CSV file."""
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.dirs['summary'], f'{self.model_id}_plot_history.csv')
        df.to_csv(csv_path, index=False)


def create_plotter_from_log(logfile, plot_dir='./plot'):
    """
    Create plots from an existing log file.

    Args:
        logfile: Path to the CSV log file
        plot_dir: Base directory for plots

    Returns:
        TrainingPlotter instance with populated history
    """
    # Parse path to extract dataset and phase
    parts = logfile.split(os.sep)
    dataset_name = None
    phase = None

    for i, part in enumerate(parts):
        if part in ['timit', 'gscdv2']:
            dataset_name = part
        if part in ['pretrain', 'retrain']:
            phase = part

    if dataset_name is None or phase is None:
        raise ValueError(f"Could not parse dataset/phase from path: {logfile}")

    # Get model_id from filename
    model_id = os.path.basename(logfile).replace('.csv', '')

    # Create plotter
    plotter = TrainingPlotter(plot_dir, model_id, dataset_name, phase)

    # Load log file
    df = pd.read_csv(logfile)

    # Populate history
    for _, row in df.iterrows():
        log_stat = row.to_dict()
        plotter.update(log_stat)

    return plotter
