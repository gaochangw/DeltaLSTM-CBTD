"""
Evaluation metrics for Google Speech Commands Dataset v2 - Keyword Spotting.
Includes accuracy, confusion matrix, per-class metrics, F1 score, FAR, EER, and AUC-ROC.
"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    roc_curve, auc, roc_auc_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# 12-class KWS labels
KWS_LABELS = ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


def gen_meter_args(args, n_classes, **kwargs):
    """Generate meter arguments."""
    dict_meter_args = {
        'n_classes': n_classes,
        'labels': KWS_LABELS[:n_classes]
    }
    return dict_meter_args


class Meter:
    """Meter class for KWS evaluation metrics."""

    def __init__(self, dict_meter_args):
        self.n_classes = dict_meter_args['n_classes']
        self.labels = dict_meter_args.get('labels', KWS_LABELS[:self.n_classes])

        # Storage for predictions, targets, and probabilities
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []  # Store probability scores for AUC-ROC

    def clear_data(self):
        """Clear accumulated data."""
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []

    def clear(self):
        """Alias for clear_data."""
        self.clear_data()

    def add_batch(self, preds, targets, probs=None):
        """Add batch predictions and targets."""
        self.all_preds.extend(preds.flatten().tolist())
        self.all_targets.extend(targets.flatten().tolist())
        if probs is not None:
            if isinstance(probs, np.ndarray):
                self.all_probs.extend(probs.tolist())
            else:
                self.all_probs.extend(probs)

    def extend_data(self, outputs, targets, target_lengths=None):
        """Extend data from batch outputs (compatibility with TIMIT interface)."""
        for output, target in zip(outputs, targets):
            if isinstance(output, np.ndarray):
                if output.ndim == 3:
                    # (B, T, C) - average over time
                    output = output.mean(axis=1)
                # Apply softmax to get probabilities
                exp_output = np.exp(output - np.max(output, axis=-1, keepdims=True))
                probs = exp_output / np.sum(exp_output, axis=-1, keepdims=True)
                preds = np.argmax(output, axis=-1)
            else:
                preds = output
                probs = None

            self.all_preds.extend(preds.flatten().tolist())
            self.all_targets.extend(target.flatten().tolist())
            if probs is not None:
                self.all_probs.extend(probs.tolist())

        return self.all_preds, self.all_targets

    def _compute_far_frr_per_class(self, targets, preds):
        """Compute FAR and FRR for each class (one-vs-rest)."""
        n_classes = self.n_classes
        far_per_class = {}
        frr_per_class = {}

        for c in range(n_classes):
            # Binary classification: class c vs all others
            true_positive = np.sum((targets == c) & (preds == c))
            false_positive = np.sum((targets != c) & (preds == c))
            true_negative = np.sum((targets != c) & (preds != c))
            false_negative = np.sum((targets == c) & (preds != c))

            # FAR = FP / (FP + TN) - rate of accepting non-target as target
            far = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0
            # FRR = FN / (FN + TP) - rate of rejecting target
            frr = false_negative / (false_negative + true_positive) if (false_negative + true_positive) > 0 else 0

            far_per_class[self.labels[c]] = far
            frr_per_class[self.labels[c]] = frr

        return far_per_class, frr_per_class

    def _compute_eer_per_class(self, targets, probs):
        """Compute EER for each class using probability scores."""
        if len(probs) == 0:
            return {label: 0.0 for label in self.labels}

        probs_array = np.array(probs)
        targets_array = np.array(targets)
        eer_per_class = {}

        for c in range(self.n_classes):
            # Binary labels for this class
            binary_targets = (targets_array == c).astype(int)

            # Skip if only one class present
            if len(np.unique(binary_targets)) < 2:
                eer_per_class[self.labels[c]] = 0.0
                continue

            # Get scores for this class
            scores = probs_array[:, c]

            try:
                fpr, tpr, thresholds = roc_curve(binary_targets, scores)
                fnr = 1 - tpr

                # Find EER - point where FPR = FNR
                eer_func = interp1d(fpr, fnr)
                eer = brentq(lambda x: eer_func(x) - x, 0.0, 1.0)
                eer_per_class[self.labels[c]] = eer
            except Exception:
                eer_per_class[self.labels[c]] = 0.0

        return eer_per_class

    def _compute_auc_roc_per_class(self, targets, probs):
        """Compute AUC-ROC for each class (one-vs-rest)."""
        if len(probs) == 0:
            return {label: 0.0 for label in self.labels}, 0.0

        probs_array = np.array(probs)
        targets_array = np.array(targets)
        auc_per_class = {}
        roc_data = {}  # Store ROC curve data for plotting

        for c in range(self.n_classes):
            # Binary labels for this class
            binary_targets = (targets_array == c).astype(int)

            # Skip if only one class present
            if len(np.unique(binary_targets)) < 2:
                auc_per_class[self.labels[c]] = 0.5
                roc_data[self.labels[c]] = {'fpr': [0, 1], 'tpr': [0, 1]}
                continue

            # Get scores for this class
            scores = probs_array[:, c]

            try:
                fpr, tpr, _ = roc_curve(binary_targets, scores)
                auc_score = auc(fpr, tpr)
                auc_per_class[self.labels[c]] = auc_score
                roc_data[self.labels[c]] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            except Exception:
                auc_per_class[self.labels[c]] = 0.5
                roc_data[self.labels[c]] = {'fpr': [0, 1], 'tpr': [0, 1]}

        # Compute macro-average AUC
        try:
            # One-hot encode targets
            targets_onehot = np.zeros((len(targets_array), self.n_classes))
            targets_onehot[np.arange(len(targets_array)), targets_array] = 1
            macro_auc = roc_auc_score(targets_onehot, probs_array, average='macro', multi_class='ovr')
        except Exception:
            macro_auc = np.mean(list(auc_per_class.values()))

        return auc_per_class, macro_auc, roc_data

    def get_metrics(self, dict_stat):
        """Calculate and return all metrics."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        # Calculate accuracy
        accuracy = np.mean(preds == targets)
        error_rate = 1 - accuracy

        # Calculate per-class accuracy
        per_class_acc = {}
        for i, label in enumerate(self.labels):
            mask = targets == i
            if np.sum(mask) > 0:
                per_class_acc[label] = np.mean(preds[mask] == targets[mask])
            else:
                per_class_acc[label] = 0.0

        # Calculate F1 score
        f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(targets, preds, average='weighted', zero_division=0)

        # Calculate FAR and FRR per class
        far_per_class, frr_per_class = self._compute_far_frr_per_class(targets, preds)

        # Calculate EER per class (requires probability scores)
        eer_per_class = self._compute_eer_per_class(targets, self.all_probs)

        # Calculate AUC-ROC per class
        auc_per_class, macro_auc, roc_data = self._compute_auc_roc_per_class(targets, self.all_probs)

        # Store metrics
        dict_stat['accuracy'] = accuracy
        dict_stat['error_rate'] = error_rate
        dict_stat['per'] = error_rate  # For compatibility with TIMIT code
        dict_stat['lr_criterion'] = error_rate  # For LR scheduler
        dict_stat['f1_macro'] = f1_macro
        dict_stat['f1_weighted'] = f1_weighted
        dict_stat['per_class_acc'] = per_class_acc

        # New metrics
        dict_stat['far_per_class'] = far_per_class
        dict_stat['frr_per_class'] = frr_per_class
        dict_stat['far_macro'] = np.mean(list(far_per_class.values()))
        dict_stat['eer_per_class'] = eer_per_class
        dict_stat['eer_macro'] = np.mean(list(eer_per_class.values()))
        dict_stat['auc_per_class'] = auc_per_class
        dict_stat['auc_macro'] = macro_auc
        dict_stat['roc_data'] = roc_data

        # Clear data after metrics calculation
        self.clear_data()

        return dict_stat

    def get_metrics_preserve(self):
        """Get metrics without clearing data."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        error_rate = 1 - np.mean(preds == targets)
        return error_rate

    def get_confusion_matrix(self):
        """Return confusion matrix."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        return confusion_matrix(targets, preds, labels=list(range(self.n_classes)))

    def get_classification_report(self):
        """Return detailed classification report."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        return classification_report(
            targets, preds,
            labels=list(range(self.n_classes)),
            target_names=self.labels,
            zero_division=0
        )

    def print_confusion_matrix(self):
        """Print formatted confusion matrix."""
        cm = self.get_confusion_matrix()
        print("\nConfusion Matrix:")
        print("-" * 80)

        # Header
        header = "True\\Pred |"
        for label in self.labels:
            header += f" {label[:6]:>6} |"
        print(header)
        print("-" * 80)

        # Rows
        for i, label in enumerate(self.labels):
            row = f"{label[:9]:>9} |"
            for j in range(len(self.labels)):
                row += f" {cm[i, j]:>6} |"
            print(row)
        print("-" * 80)

    def print_per_class_accuracy(self, dict_stat):
        """Print per-class accuracy."""
        print("\nPer-class Accuracy:")
        print("-" * 40)
        per_class = dict_stat.get('per_class_acc', {})
        for label, acc in per_class.items():
            print(f"  {label:>10}: {acc * 100:.2f}%")
        print("-" * 40)


def calculate_error_rates(target_labels, guessed_labels):
    """Calculate error rate (for compatibility with TIMIT)."""
    if isinstance(target_labels, list):
        targets = np.array(target_labels)
        preds = np.array(guessed_labels)
    else:
        targets = target_labels
        preds = guessed_labels

    error_rate = 1 - np.mean(preds == targets)
    return error_rate
