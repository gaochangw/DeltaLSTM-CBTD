"""
Evaluation metrics for 3ET+ Event-Based Eye Tracking - Regression task.
Computes Euclidean distance and pixel-tolerance accuracy (p-accuracy).
"""
import numpy as np
import torch


def gen_meter_args(args, n_classes, **kwargs):
    """Generate meter arguments."""
    spatial_factor = getattr(args, 'spatial_factor', 0.125)
    sensor_width = getattr(args, 'sensor_width', 640)
    sensor_height = getattr(args, 'sensor_height', 480)
    pixel_tolerances_str = getattr(args, 'pixel_tolerances', '5,10,15')
    pixel_tolerances = [int(x) for x in pixel_tolerances_str.split(',')]

    best_model_criterion = getattr(args, 'best_model_criterion', 'euclidean_dist')

    dict_meter_args = {
        'n_classes': n_classes,
        'sensor_width': sensor_width,
        'sensor_height': sensor_height,
        'spatial_factor': spatial_factor,
        'ds_width': int(sensor_width * spatial_factor),
        'ds_height': int(sensor_height * spatial_factor),
        'pixel_tolerances': pixel_tolerances,
        'best_model_criterion': best_model_criterion,
    }
    return dict_meter_args


class Meter:
    """Meter class for 3ET+ regression evaluation metrics."""

    def __init__(self, dict_meter_args):
        self.n_classes = dict_meter_args['n_classes']
        self.ds_width = dict_meter_args['ds_width']
        self.ds_height = dict_meter_args['ds_height']
        self.pixel_tolerances = dict_meter_args['pixel_tolerances']
        self.best_model_criterion = dict_meter_args.get('best_model_criterion', 'euclidean_dist')

        # Storage for predictions and targets
        self.all_preds = []    # List of (B, T, 2) tensors
        self.all_targets = []  # List of (B, T, 3) tensors

    def clear_data(self):
        """Clear accumulated data."""
        self.all_preds = []
        self.all_targets = []

    def clear(self):
        """Alias for clear_data."""
        self.clear_data()

    def add_batch(self, predictions, targets_full):
        """Add batch predictions and targets.

        Args:
            predictions: (B, T, 2) tensor — normalized x, y
            targets_full: (B, T, 3) tensor — normalized x, y, close_flag
        """
        self.all_preds.append(predictions)
        self.all_targets.append(targets_full)

    def get_metrics(self, dict_stat):
        """Calculate and return regression metrics."""
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.all_preds, dim=0)     # (N, T, 2)
        all_targets = torch.cat(self.all_targets, dim=0)  # (N, T, 3)

        # Flatten N and T dimensions
        preds_flat = all_preds.reshape(-1, 2)           # (N*T, 2)
        targets_flat = all_targets.reshape(-1, 3)       # (N*T, 3)

        # Compute distances in pixel space (rescale from [0,1] to sensor coords)
        dis = targets_flat[:, :2] - preds_flat
        dis[:, 0] *= self.ds_width
        dis[:, 1] *= self.ds_height
        dist = torch.norm(dis, dim=-1)  # (N*T,)

        # Mean Euclidean distance
        euclidean_dist = dist.mean().item()

        # P-accuracy: % of predictions within pixel tolerance
        p_acc = {}
        for p_tol in self.pixel_tolerances:
            p_acc[f'p{p_tol}'] = (dist < p_tol).float().mean().item()

        # P-accuracy excluding closed eyes
        eye_closed = targets_flat[:, 2]
        open_mask = eye_closed == 0
        n_open = open_mask.sum().item()

        p_acc_open = {}
        if n_open > 0:
            dist_open = dist[open_mask]
            euclidean_dist_open = dist_open.mean().item()
            for p_tol in self.pixel_tolerances:
                p_acc_open[f'p{p_tol}_open'] = (dist_open < p_tol).float().mean().item()
        else:
            euclidean_dist_open = euclidean_dist
            for p_tol in self.pixel_tolerances:
                p_acc_open[f'p{p_tol}_open'] = 0.0

        # Store metrics
        dict_stat['euclidean_dist'] = euclidean_dist
        dict_stat['euclidean_dist_open'] = euclidean_dist_open
        dict_stat['per'] = euclidean_dist  # For compatibility (lower is better)
        dict_stat['accuracy'] = p_acc.get('p10', 0.0)  # For logging compatibility

        # lr_criterion: supports best_model_criterion for p-accuracy metrics
        # LR scheduler uses mode='min', so negate p-accuracy (higher is better -> lower is better)
        best_criterion = self.best_model_criterion
        if best_criterion in ('p3', 'p5', 'p10', 'p15'):
            dict_stat['lr_criterion'] = -p_acc.get(best_criterion, 0.0)
        else:
            dict_stat['lr_criterion'] = euclidean_dist  # For LR scheduler (mode='min')

        # Add p-accuracy metrics
        for k, v in p_acc.items():
            dict_stat[k] = v
        for k, v in p_acc_open.items():
            dict_stat[k] = v

        # Clear data after metrics calculation
        self.clear_data()

        return dict_stat

    def get_metrics_preserve(self):
        """Get metrics without clearing data."""
        all_preds = torch.cat(self.all_preds, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)

        preds_flat = all_preds.reshape(-1, 2)
        targets_flat = all_targets.reshape(-1, 3)

        dis = targets_flat[:, :2] - preds_flat
        dis[:, 0] *= self.ds_width
        dis[:, 1] *= self.ds_height
        dist = torch.norm(dis, dim=-1)

        return dist.mean().item()
