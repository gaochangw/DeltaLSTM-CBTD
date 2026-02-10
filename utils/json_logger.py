"""
JSON Logger for DeltaLSTM-CBTD
Creates structured JSON files that are easy to parse for statistics and plotting.
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class JsonLogger:
    """
    Structured JSON logger for easy parsing and analysis.

    Output structure:
    {
        "metadata": {
            "model_id": "...",
            "dataset": "...",
            "phase": "...",
            "timestamp": "...",
            "version": "1.0"
        },
        "config": {
            "model": {...},
            "training": {...},
            "quantization": {...},
            "delta_lstm": {...}  # if applicable
        },
        "complexity": {
            "params": {...},
            "flops": {...},
            "memory": {...}
        },
        "training_history": [
            {"epoch": 1, "train": {...}, "val": {...}, "test": {...}, "sparsity": {...}},
            ...
        ],
        "best_results": {
            "epoch": ...,
            "val": {...},
            "test": {...}
        },
        "final_sparsity": {
            "weight": {...},
            "temporal": {...}  # if DeltaLSTM
        }
    }
    """

    def __init__(self, log_dir: str, model_id: str, dataset_name: str, phase: str):
        """
        Initialize the JSON logger.

        Args:
            log_dir: Directory to save the JSON file
            model_id: Model identifier
            dataset_name: Dataset name (timit, gscdv2)
            phase: Training phase (pretrain, retrain)
        """
        self.log_dir = log_dir
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.phase = phase

        # Create directory if needed
        os.makedirs(log_dir, exist_ok=True)

        # JSON file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = os.path.join(
            log_dir, f"{dataset_name}_{phase}_{timestamp}.json"
        )

        # Initialize data structure
        self.data = {
            "metadata": {
                "model_id": model_id,
                "dataset": dataset_name,
                "phase": phase,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "config": {
                "model": {},
                "training": {},
                "quantization": {},
                "delta_lstm": {}
            },
            "complexity": {
                "params": {},
                "flops": {},
                "memory": {}
            },
            "training_history": [],
            "best_results": {},
            "final_sparsity": {
                "weight": {},
                "temporal": {}
            }
        }

    def set_config(self, args, n_features: int, n_classes: int):
        """Store model and training configuration."""
        self.data["config"]["model"] = {
            "type": args.hid_type,
            "hidden_layers": args.hid_layers,
            "hidden_size": args.hid_size,
            "input_features": n_features,
            "output_classes": n_classes,
            "fc_extra_size": args.fc_extra_size,
            "dropout": args.hid_dropout
        }

        self.data["config"]["training"] = {
            "batch_size": args.batch_size,
            "epochs": args.n_epochs,
            "learning_rate": args.lr,
            "learning_rate_end": getattr(args, 'lr_end', None),
            "optimizer": getattr(args, 'opt', 'ADAMW'),
            "weight_decay": getattr(args, 'weight_decay', 0),
            "loss_function": args.loss,
            "seed": args.seed
        }

        self.data["config"]["quantization"] = {
            "activations": {
                "enabled": args.qa,
                "int_bits": args.aqi,
                "frac_bits": args.aqf
            },
            "weights": {
                "enabled": args.qw,
                "int_bits": args.wqi,
                "frac_bits": args.wqf
            }
        }

        # DeltaLSTM specific config
        if 'Delta' in args.hid_type:
            self.data["config"]["delta_lstm"] = {
                "threshold_x": args.th_x,
                "threshold_h": args.th_h,
                "target_sparsity_rnn": args.gamma_rnn,
                "target_sparsity_fc": args.gamma_fc,
                "balance_pe": getattr(args, 'balance_pe', False),
                "num_arrays": args.num_array,
                "pes_per_array": args.num_array_pe,
                "cbtd_enabled": getattr(args, 'cbtd', False),
                "msq_lambda": getattr(args, 'msq_lambda', 0),
                "msq_k": getattr(args, 'msq_k', 2),
                "msq_target": getattr(args, 'msq_target', 'rnn'),
                "msq_anneal": getattr(args, 'msq_anneal', 0),
                "bitwave": getattr(args, 'bitwave', 0)
            }

    def set_complexity(self, model_stats: Dict[str, Any]):
        """Store model complexity statistics."""
        self.data["complexity"]["params"] = {
            "total": model_stats.get('n_params_total', model_stats.get('total', 0)),
            "trainable": model_stats.get('n_params_trainable', model_stats.get('trainable', 0)),
            "non_trainable": model_stats.get('n_params_total', 0) - model_stats.get('n_params_trainable', 0)
        }

        self.data["complexity"]["flops"] = {
            "total": model_stats.get('flops', 0),
            "madds": model_stats.get('madds', 0)
        }

        self.data["complexity"]["memory"] = {
            "parameters_mb": model_stats.get('memory_params_mb', model_stats.get('param_memory_mb', 0)),
            "activations_mb": model_stats.get('memory_activations_mb', model_stats.get('activation_memory_mb', 0)),
            "total_mb": model_stats.get('memory_total_mb', model_stats.get('total_memory_mb', 0))
        }

        self.data["complexity"]["input"] = {
            "sequence_length": model_stats.get('seq_len', 0),
            "features": model_stats.get('n_features', 0)
        }

        # Per-layer breakdown
        if 'layer_breakdown' in model_stats:
            self.data["complexity"]["layer_breakdown"] = model_stats['layer_breakdown']

    def add_epoch(self, log_stat: Dict[str, Any]):
        """Add epoch data to training history."""
        epoch = log_stat.get('EPOCH', len(self.training_history) + 1)

        # Extract metrics by category
        epoch_data = {
            "epoch": epoch,
            "time": log_stat.get('TIME_CURR', ''),
            "learning_rate": log_stat.get('LR_CURR', 0),
            "alpha": log_stat.get('ALPHA', 0)
        }

        # Training metrics
        train_metrics = {}
        for key, value in log_stat.items():
            if key.startswith('TRAIN_'):
                metric_name = key[6:].lower()  # Remove 'TRAIN_' prefix
                train_metrics[metric_name] = self._convert_value(value)
        if train_metrics:
            epoch_data["train"] = train_metrics

        # Validation metrics
        val_metrics = {}
        for key, value in log_stat.items():
            if key.startswith('VAL_'):
                metric_name = key[4:].lower()  # Remove 'VAL_' prefix
                val_metrics[metric_name] = self._convert_value(value)
        if val_metrics:
            epoch_data["val"] = val_metrics

        # Test metrics
        test_metrics = {}
        for key, value in log_stat.items():
            if key.startswith('TEST_'):
                metric_name = key[5:].lower()  # Remove 'TEST_' prefix
                test_metrics[metric_name] = self._convert_value(value)
        if test_metrics:
            epoch_data["test"] = test_metrics

        # Sparsity metrics
        sparsity = {}
        if 'SP_W_RNN' in log_stat:
            sparsity["weight_rnn"] = self._convert_value(log_stat['SP_W_RNN'])
        if 'SP_W_FC' in log_stat:
            sparsity["weight_fc"] = self._convert_value(log_stat['SP_W_FC'])

        # Temporal sparsity (from val or test)
        sp_dx = log_stat.get('VAL_SP_DX', log_stat.get('TEST_SP_DX'))
        sp_dh = log_stat.get('VAL_SP_DH', log_stat.get('TEST_SP_DH'))
        sp_dxh = log_stat.get('VAL_SP_DXH', log_stat.get('TEST_SP_DXH'))

        if sp_dx is not None:
            sparsity["temporal_dx"] = self._convert_value(sp_dx)
        if sp_dh is not None:
            sparsity["temporal_dh"] = self._convert_value(sp_dh)
        if sp_dxh is not None:
            sparsity["temporal_dxh"] = self._convert_value(sp_dxh)

        # Bit-level sparsity (two's complement)
        if 'SP_BIT_RNN' in log_stat:
            sparsity["bit_rnn"] = self._convert_value(log_stat['SP_BIT_RNN'])
        if 'SP_BIT_NZ_RNN' in log_stat:
            sparsity["bit_nz_rnn"] = self._convert_value(log_stat['SP_BIT_NZ_RNN'])
        # Bit-level sparsity (sign-magnitude / BitWave)
        if 'SP_BIT_SM_RNN' in log_stat:
            sparsity["bit_sm_rnn"] = self._convert_value(log_stat['SP_BIT_SM_RNN'])
        if 'SP_BIT_NZ_SM_RNN' in log_stat:
            sparsity["bit_nz_sm_rnn"] = self._convert_value(log_stat['SP_BIT_NZ_SM_RNN'])
        if 'BITWAVE' in log_stat:
            sparsity["bitwave"] = log_stat['BITWAVE']

        if sparsity:
            epoch_data["sparsity"] = sparsity

        self.data["training_history"].append(epoch_data)

    @property
    def training_history(self) -> List[Dict]:
        """Get training history."""
        return self.data["training_history"]

    def _convert_value(self, value: Any) -> Any:
        """Convert value to JSON-serializable type."""
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._convert_value(v) for v in value]
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def set_best_results(self, best_epoch: int, best_val_metric: float,
                         metric_name: str = 'accuracy'):
        """Set best results from training."""
        self.data["best_results"] = {
            "epoch": best_epoch,
            "metric_name": metric_name,
            "best_value": self._convert_value(best_val_metric)
        }

        # Find the best epoch data
        for epoch_data in self.data["training_history"]:
            if epoch_data.get("epoch") == best_epoch:
                if "val" in epoch_data:
                    self.data["best_results"]["val"] = epoch_data["val"]
                if "test" in epoch_data:
                    self.data["best_results"]["test"] = epoch_data["test"]
                if "sparsity" in epoch_data:
                    self.data["best_results"]["sparsity"] = epoch_data["sparsity"]
                break

    def set_final_sparsity(self, weight_sparsity: Dict[str, float],
                           temporal_sparsity: Optional[Dict[str, float]] = None):
        """Set final sparsity statistics."""
        self.data["final_sparsity"]["weight"] = {
            k: self._convert_value(v) for k, v in weight_sparsity.items()
        }

        if temporal_sparsity:
            self.data["final_sparsity"]["temporal"] = {
                k: self._convert_value(v) for k, v in temporal_sparsity.items()
            }

    def save(self) -> str:
        """Save JSON file and return path."""
        # Update metadata with completion time
        self.data["metadata"]["completed_at"] = datetime.now().isoformat()

        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, cls=NumpyEncoder)

        return self.json_path

    def get_data(self) -> Dict[str, Any]:
        """Get the full data dictionary."""
        return self.data


def load_json_log(json_path: str) -> Dict[str, Any]:
    """
    Load a JSON log file.

    Args:
        json_path: Path to JSON log file

    Returns:
        Dictionary with log data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_training_dataframe(json_path: str):
    """
    Convert JSON log training history to pandas DataFrame.

    Args:
        json_path: Path to JSON log file

    Returns:
        pandas DataFrame with training history
    """
    import pandas as pd

    data = load_json_log(json_path)
    history = data.get("training_history", [])

    # Flatten nested dictionaries
    flat_records = []
    for epoch_data in history:
        flat = {"epoch": epoch_data.get("epoch")}

        for key, value in epoch_data.items():
            if key == "epoch":
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            flat[f"{key}_{sub_key}_{sub_sub_key}"] = sub_sub_value
                    else:
                        flat[f"{key}_{sub_key}"] = sub_value
            else:
                flat[key] = value

        flat_records.append(flat)

    return pd.DataFrame(flat_records)


def compare_runs(json_paths: List[str]) -> Dict[str, Any]:
    """
    Compare multiple training runs.

    Args:
        json_paths: List of paths to JSON log files

    Returns:
        Dictionary with comparison data
    """
    comparison = {
        "runs": [],
        "metrics_comparison": {}
    }

    for path in json_paths:
        data = load_json_log(path)
        run_info = {
            "path": path,
            "model_id": data["metadata"]["model_id"],
            "config": data["config"],
            "best_results": data.get("best_results", {}),
            "final_sparsity": data.get("final_sparsity", {})
        }
        comparison["runs"].append(run_info)

    return comparison
