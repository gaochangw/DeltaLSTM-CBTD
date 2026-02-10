# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeltaLSTM-CBTD is a PyTorch research implementation of sparsity-exploiting LSTMs with Column-Balanced Targeted Dropout (CBTD) for efficient neural network inference on resource-constrained hardware. The project supports:
- **TIMIT**: Phoneme recognition with CTC loss, 2-layer 1024-unit LSTM
- **GSCDv2**: 12-class keyword spotting (yes, no, up, down, left, right, on, off, stop, go, _silence_, _unknown_) with cross-entropy loss, 1-layer 128-unit LSTM
- **3ET+ (threeet)**: Event-based eye tracking with pupil center (x,y) regression using weighted MSE loss, 1-layer 256-unit LSTM (also supports RVT and JaneEyeNet architectures). Uses CNN spatial frontend for voxel grid processing (240x180 → 80x60).

TIMIT and GSCDv2 use fixed-point quantization and hardware-aware workload balancing. 3ET+ extends the framework to regression tasks with event camera data.

## Environment Setup

Requires Miniconda with Python 3.13 and PyTorch (latest stable) installed in the base environment. There is no `requirements.txt` at the project root; install dependencies manually:

```bash
pip install numpy matplotlib pandas tqdm h5py scipy scikit-learn tensorboard editdistance sphfile rich seaborn openpyxl
```

There is no test suite, linter configuration, or CI/CD pipeline. Validation is done by running the train/test pipeline and checking metrics.

## Build/Run Commands

The project uses a 4-step pipeline executed via `main.py`. Switch datasets via `--dataset_name {timit|gscdv2|threeet}`:

### TIMIT (Phoneme Recognition)
```bash
python main.py --dataset_name timit --step data_prep
python main.py --dataset_name timit --step feat_ext
python main.py --dataset_name timit --step train --phase pretrain --hid_type LSTM
python main.py --dataset_name timit --step test --phase pretrain --hid_type LSTM
```

### GSCDv2 (Keyword Spotting)
```bash
python main.py --dataset_name gscdv2 --step data_prep --data_dir /path/to/gscdv2
python main.py --dataset_name gscdv2 --step feat_ext
python main.py --dataset_name gscdv2 --step train --phase pretrain --hid_type LSTM --gpu_device 0
python main.py --dataset_name gscdv2 --step test --phase pretrain --hid_type LSTM --gpu_device 0
```

### 3ET+ (Event-Based Eye Tracking)
```bash
python main.py --dataset_name threeet --step data_prep --data_dir /path/to/3et_data
python main.py --dataset_name threeet --step feat_ext
python main.py --dataset_name threeet --step train --phase pretrain --hid_type LSTM --gpu_device 0
python main.py --dataset_name threeet --step test --phase pretrain --hid_type LSTM --gpu_device 0
```

Alternative architectures for 3ET+:
```bash
# RVT (Recurrent Vision Transformer)
python main.py --dataset_name threeet --step train --phase pretrain --hid_type RVT --gpu_device 0
# JaneEyeNet (simplified RVT with 3-conv blocks, no attention)
python main.py --dataset_name threeet --step train --phase pretrain --hid_type JaneEyeNet --gpu_device 0
```

### DeltaLSTM Retraining (both datasets)
```bash
python main.py --dataset_name {timit|gscdv2} --step train --phase retrain --hid_type DeltaLSTM --hid_type_pretrain LSTM
python main.py --dataset_name {timit|gscdv2} --step test --phase retrain --hid_type DeltaLSTM
```

### MSQ + BitWave (Bit Sparsification)
```bash
# Pretrain LSTM with MSQ LSB regularization (bit sparsification during pretrain)
python main.py --dataset_name gscdv2 --step train --phase pretrain --hid_type LSTM --msq_lambda 1.0 --msq_k 2

# Retrain as DeltaLSTM (MSQ auto-disabled; temporal sparsity only)
python main.py --dataset_name gscdv2 --step train --phase retrain --hid_type DeltaLSTM --hid_type_pretrain LSTM

# Test with BitWave sign-magnitude analysis
python main.py --dataset_name gscdv2 --step test --phase retrain --hid_type DeltaLSTM --bitwave 1
```

### Hyperparameter Study
```bash
python hyperparam_study.py --quick  # Quick test with reduced parameters
python hyperparam_study.py --pretrain_epochs 50 --retrain_epochs 5  # Full study
```

### Design Space Exploration (DSE)
```bash
# Quick test run (~20 min, 2 gammas × 2 thresholds, 2 pretrain + 1 retrain epochs)
bash run_dse.sh

# Save full console log
bash run_dse.sh 2>&1 | tee dse_log.txt

# Individual DSE tools
python merge_logs.py --dataset_name gscdv2 --phase retrain --source test --sort_by TEST_ACCURACY
python plot_dse.py --dataset_name gscdv2 --output_dir ./plot/dse
```

### Benchmarking
```bash
python bench.py                  # Quick forward pass benchmark
python bench_profile.py          # PyTorch profiler analysis
python benchmark_deltalstm.py    # DeltaLSTM-specific benchmarking
python benchmark_quick.py        # Quick benchmark variant
python benchmark_focused.py      # Focused benchmark variant
```

## Architecture

### Entry Point and Pipeline
- `main.py`: Orchestrates the 4-step pipeline (data_prep → feat_ext → train → test) via a `Project` class that tracks step index
- Uses `importlib.import_module('modules.' + args.dataset_name + '.log')` to dynamically load dataset-specific modules
- `--run_through` flag runs all steps sequentially from the specified step onward

### Core Model Components
- `model.py`: PyTorch `Model` class supporting LSTM, GRU, and DeltaLSTM with optional FC layers and quantization
- `nnmodels/deltalstm.py`: Optimized `DeltaRNN` class implementing sparse delta-based computation
  - Key methods: `deltalstm_layer_forward()`, `balance_workload()`, `adapt_net_to_hw()`, `update_stats()`
  - Tracks temporal sparsity via delta vectors between timesteps
  - Implements CBTD for hardware PE workload balancing
- `nnmodels/deltalstm_original.py`: Backup of unoptimized version for reference
- `nnmodels/deltalstm_optimized.py`: Alternate optimized variant
- `nnmodels/rvt.py`: Recurrent Vision Transformer for event-based eye tracking (3ET+ only). Conv → LayerNorm → FullSelfAttention → MLP → DWSConvLSTM2d per stage
- `nnmodels/janeyenet.py`: JaneEye-Net simplified RVT with 3-conv blocks, no attention, forget-gate-only LSTM (3ET+ only)

### Dataset-Specific Modules (`modules/{dataset_name}/`)
Each dataset has its own module directory with consistent interfaces:
- `arguments.py`: Hyperparameters for training, quantization, and hardware configuration
- `data_collector.py`, `feat_extractor.py`, `dataloader.py`: Data pipeline
- `train_func.py`, `metric.py`, `log.py`: Training, evaluation, and logging (GSCDv2 uses rich formatting)

### Utilities (`utils/`)
- `util.py`: Core utilities including `quantize_tensor()`, `quantize_array()`, gradient-preserving quantization autograd functions (`GradPreserveRound`, `GradPreserveFloor`), `cbtd()` for Column-Balanced Targeted Dropout
- `model_stats.py`: Model complexity analyzer using torchstat2 - provides FLOPs, MADDs, memory usage, parameter count, and layer-wise breakdown with rich console output
- `excel_logger.py`: Comprehensive Excel logger that creates beautifully formatted multi-sheet xlsx files with model config, complexity, training history, metrics, sparsity analysis, and DeltaLSTM-specific stats
- `msq_regularizer.py`: MSQ-style sinusoidal LSB regularization (`compute_lsb_penalty()`) that encourages lower k bits of quantized weights to zero, increasing bit-level sparsity
- `sparsity_analysis.py`: Bit-level sparsity analysis with 5 speedup scenarios (`analyze_sparsity_scenarios()`). Supports two's complement and sign-magnitude (BitWave) representations
- `weighted_mse.py`: Weighted MSE/RMSE loss for regression tasks with aspect ratio compensation (3ET+ eye tracking)
- `feature/filter_bank.py`: Audio filter bank computation for MFCC-like features
- `training_plotter.py`: Training log visualization

### External Libraries (`torchstat2/`)
- Local copy of torchstat2 for model analysis (FLOPs, MADDs, memory, throughput benchmarking)
- Usage: `from torchstat2 import stat, throughput, stat_with_throughput`

### Research Tools
- `hyperparam_study.py`: Grid search over target sparsity (gamma_rnn) and delta thresholds (th_x, th_h), generates Excel reports and publication-quality plots

### Design Space Exploration Tools
- `run_dse.sh`: Automated bash script that sweeps over weight sparsity (gamma_rnn), temporal sparsity (th_x/th_h), and bit sparsity (msq_lambda) in a 4-phase pipeline: pretrain LSTM → retrain DeltaLSTM → merge logs → generate plots. All sweep parameters are tunable at the top of the script.
- `merge_logs.py`: Standalone CLI script that discovers and merges CSV/JSON/XLSX logs from multiple training runs into consolidated files. Parses model IDs to extract hyperparameters as `HP_*` columns. Outputs unified CSV, JSON, and XLSX to `./log/{dataset}/{phase}/merged/`.
- `plot_dse.py`: Reads merged CSVs from `merge_logs.py` and generates 10+ publication-quality plots across 7 categories: accuracy landscape, sparsity decomposition, speedup analysis, accuracy-speedup tradeoff, bit-level sparsity, summary dashboard, and pretrain-vs-retrain comparison. Outputs PNG (300dpi) and PDF to `./plot/dse/`.

## Key Arguments

Dataset: `--dataset_name {timit|gscdv2|threeet}`, `--data_dir`, `--gpu_device`

Training: `--hid_type {LSTM|GRU|DeltaLSTM|RVT|JaneEyeNet}`, `--hid_size`, `--hid_layers`, `--batch_size`, `--lr`, `--n_epochs`

Quantization: `--qa` (quantize activations), `--qw` (quantize weights), `--aqi/aqf` (activation int/frac bits), `--wqi/wqf` (weight int/frac bits)

DeltaLSTM: `--th_x`, `--th_h` (sparsity thresholds 0.0-1.0), `--balance_pe` (enable CBTD), `--gamma_rnn/gamma_fc` (target sparsity 0.0-0.95)

MSQ Bit Sparsification: `--msq_lambda` (0=disabled, e.g. 1.0), `--msq_k` (LSBs to penalize, default 2), `--msq_target` (rnn|fc|all), `--msq_anneal` (0=constant, 1=linear ramp). Only active during pretrain phase.

BitWave: `--bitwave` (0=two's complement, 1=sign-magnitude for bit sparsity analysis)

3ET+: `--cnn_frontend` (enable CNN spatial extractor), `--spatial_factor` (downsampling, default 1/3), `--n_time_bins` (voxel temporal bins), `--seq_length` (frames per sequence), `--pixel_tolerances` (p-accuracy thresholds), `--best_model_criterion` (euclidean_dist|p3|p5|p10), `--tbptt` (truncated BPTT window, 0=disabled)

Hardware: `--num_array` (default 8), `--num_array_pe` (default 64)

## Logging Outputs

Training and testing automatically generate multiple log files:
- **CSV logs**: Per-epoch metrics in `./log/{dataset}/{phase}/`
- **JSON logs**: Structured logs for easy parsing (statistics and plotting)
  - Hierarchical structure with metadata, config, complexity, training_history
  - Helper functions: `load_json_log()`, `get_training_dataframe()`, `compare_runs()`
  - Includes temporal sparsity: `sp_dx` (Δx), `sp_dh` (Δh), `sp_dxh` (combined Δxh)
- **Excel logs**: Comprehensive multi-sheet xlsx files with formatted tables
  - Summary: Best results and overview
  - Configuration: All model and training parameters
  - Complexity: FLOPs, MADDs, memory, layer breakdown
  - Training History: All epoch metrics with loss chart
  - Metrics Detail: Accuracy and error metrics
  - Sparsity: Weight and temporal sparsity (Δx, Δh, Δxh) over epochs
  - DeltaLSTM: Delta-specific configuration and benefits
- **Plots**: Training curves saved to `./plot/`
- **Console output**: Rich-formatted tables with model analysis

### Parsing JSON Logs
```python
from utils.json_logger import load_json_log, get_training_dataframe

# Load full data
data = load_json_log('path/to/log.json')
print(data['complexity']['params']['total'])
print(data['training_history'][-1]['sparsity']['temporal_dxh'])

# Convert to pandas DataFrame
df = get_training_dataframe('path/to/log.json')
df.plot(x='epoch', y=['val_accuracy', 'sparsity_temporal_dxh'])
```

## Training Flow Details

### Quantization Timing
During **training**, weight quantization is applied dynamically after each epoch via `process_network(args, net, alpha)` in each dataset's `train_func.py`, NOT at initialization. Activation quantization is disabled during training (`net.set_qa_fc_final(0)`) and enabled during evaluation (`net.set_qa_fc_final(args.qa_fc_final)`). This ensures gradients flow through unquantized weights.

### Alpha Annealing (Two-Phase Training)
- **Pretrain**: `alpha = min(epoch / (alpha_anneal_epoch - 1), 1.0)` ramps CBTD drop probability from 0→1 over `--alpha_anneal_epoch` (default 30) epochs. LR scheduling is deferred until after annealing completes.
- **Retrain**: `alpha = 1` always. LR scheduling starts immediately. MSQ is auto-disabled (retrain focuses on temporal sparsity via DeltaLSTM).

### Dataset Module Interface Contract
Each `modules/{dataset}/train_func.py` implements 4 functions loaded dynamically:
- `get_batch_data(args, dict_batch_array)` — moves tensors to device. Note: TIMIT uses `.cuda()`, GSCDv2 uses `.to(device)`. New datasets should use `.to(device)`.
- `forward_propagation(net, dict_batch_tensor)` — returns `(out_final, state, reg)`. GSCDv2 applies mean temporal pooling before CE loss; TIMIT passes frame-level output to CTC; 3ET+ returns per-frame (x,y) predictions without pooling.
- `calculate_loss(loss_fn, net_out, dict_targets, reg, beta)` — dataset-specific loss handling
- `add_meter_data(args, meter, dict_meter_data)` — updates metric meter with batch results

Variable-length sequences: `feature_lengths` tensors flow through `get_batch_data()` → `forward_propagation()` → `model.forward()` for DeltaRNN delta masking and CTC loss computation.

### Pretrain Model Loading for Retrain
When retrain starts, loads from `./save/{dataset}/pretrain/{pretrain_model_id}.pt` unless `--pretrain_model` overrides. `util.load_model()` handles LSTM↔DeltaLSTM parameter mapping (concatenating `weight_ih`/`weight_hh`). Network is NOT re-initialized during retrain.

## Performance Notes

The DeltaLSTM implementation (`nnmodels/deltalstm.py`) is optimized to avoid GPU→CPU synchronization in the forward loop:
- No `.cpu()` calls inside timestep loops
- `reg` tensor stays on GPU device using `input.new_zeros(())`
- Preallocated output tensors with `input.new_empty()` instead of Python list growth
- Reused threshold masks to avoid redundant `abs()` computations
- ~10% speedup over original implementation with identical numerical results

## Design Space Exploration (DSE) Tutorial

The DSE pipeline automates a systematic sweep over the three sparsity dimensions — weight, temporal, and bit-level — to map the full accuracy–sparsity–speedup design space.

### Three Sparsity Dimensions

| Sparsity Type | Mechanism | Key Arguments | When Active |
|---|---|---|---|
| **Weight** | CBTD targeted dropout zeros entire columns | `--gamma_rnn`, `--gamma_fc` (0.0–0.95) | Pretrain (annealed via alpha) |
| **Temporal** | DeltaLSTM skips unchanged activations | `--th_x`, `--th_h` (0.0–1.0) | Retrain + test |
| **Bit-level** | MSQ regularization zeros LSBs of weights | `--msq_lambda`, `--msq_k` | Pretrain only (auto-disabled in retrain) |

These compose multiplicatively for theoretical speedup:
- Scenario 1: Weight only → `1 / (1 - sp_w)`
- Scenario 2: Temporal only → `1 / (1 - sp_dxh)`
- Scenario 3: Bit-level only → `1 / (1 - bit_sp)`
- Scenario 4: Weight + Temporal → `1 / ((1 - sp_w) * (1 - sp_dxh))`
- Scenario 5: All three → `1 / ((1 - sp_w) * (1 - sp_dxh) * (1 - bit_sp_nz))`

### Prerequisites

Run data preparation and feature extraction once before DSE:

```bash
python main.py --dataset_name gscdv2 --step data_prep --data_dir /path/to/speech_commands_v0.02
python main.py --dataset_name gscdv2 --step feat_ext
```

### Quick Test Run

The default `run_dse.sh` settings run a minimal sweep for fast verification (~20 min on RTX 4090):

```bash
bash run_dse.sh
```

Default quick-test grid: 2 gammas (0.0, 0.94) × 2 thresholds (0.0, 0.25) × 1 MSQ lambda (0.0) = 4 retrain configs, 12 total steps (train+test for each).

### Full Exploration Run

Edit the tunable parameters at the top of `run_dse.sh`:

```bash
# In run_dse.sh, change to:
PRETRAIN_EPOCHS=50             # Full training
RETRAIN_EPOCHS=5               # Full retraining
ALPHA_ANNEAL_EPOCH=30          # Full CBTD annealing (must be ≥ 2)

GAMMAS="0.0 0.5 0.8 0.9 0.94"           # 5 weight sparsity levels
THRESHOLDS="0.0 0.125 0.25 0.5"         # 4 temporal sparsity levels
MSQ_LAMBDAS="0.0 1.0"                   # With and without bit sparsification
```

Full grid: 10 pretrain (5 gammas × 2 MSQ) × 4 thresholds = 40 retrain configs, 100 total steps.

Then run:
```bash
bash run_dse.sh 2>&1 | tee dse_log.txt
```

### Pipeline Phases

`run_dse.sh` executes 4 phases automatically:

1. **Phase 1 — Pretrain LSTM**: Sweeps gamma × msq_lambda. Each config trains an LSTM with CBTD weight sparsification (and optionally MSQ bit sparsification), then tests.
2. **Phase 2 — Retrain DeltaLSTM**: For each pretrained model, sweeps delta thresholds. Loads the pretrained LSTM weights into a DeltaLSTM and retrains with temporal sparsity, then tests.
3. **Phase 3 — Merge Logs**: Runs `merge_logs.py` four times to consolidate results (pretrain best/test, retrain best/test) into `./log/{dataset}/{phase}/merged/`.
4. **Phase 4 — Plot Results**: Runs `plot_dse.py` to generate all plots from merged CSVs.

### Important Constraints

- **ALPHA_ANNEAL_EPOCH must be ≥ 2**: Setting AAE=1 causes a division-by-zero error in `steps/train.py:299` (`alpha = min(epoch / (alpha_anneal_epoch - 1), 1.0)`).
- **PRETRAIN_EPOCHS should be ≥ ALPHA_ANNEAL_EPOCH**: Otherwise CBTD annealing never reaches full sparsity.
- **MSQ is auto-disabled during retrain**: The retrain phase focuses only on temporal sparsity via DeltaLSTM.

### Using merge_logs.py Standalone

```bash
# Merge retrain test results, sorted by accuracy
python merge_logs.py --dataset_name gscdv2 --phase retrain --source test --sort_by TEST_ACCURACY

# Merge pretrain validation-best results
python merge_logs.py --dataset_name gscdv2 --phase pretrain --source best --sort_by VAL_ACCURACY

# Keep per-class columns (dropped by default to reduce bloat)
python merge_logs.py --dataset_name gscdv2 --phase retrain --source test --keep_per_class

# Skip JSON/XLSX, only produce CSV
python merge_logs.py --dataset_name gscdv2 --phase retrain --source test --no_json --no_xlsx
```

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | required | `timit` or `gscdv2` |
| `--phase` | required | `pretrain` or `retrain` |
| `--source` | `best` | Which CSVs to merge: `best` (validation-best epoch), `test` (test results), or `hist` (full history) |
| `--output_dir` | auto | Default: `./log/{dataset}/{phase}/merged/` |
| `--sort_by` | None | Column to sort by (e.g., `TEST_ACCURACY`, `VAL_ACCURACY`) |
| `--keep_per_class` | off | Keep per-class dict columns (dropped by default) |
| `--no_json` | off | Skip JSON merge |
| `--no_xlsx` | off | Skip XLSX summary |
| `--log_dir` | `./log` | Root log directory |

Output files are timestamped: `merged_{source}_{TIMESTAMP}.{csv,json,xlsx}`

### Using plot_dse.py Standalone

```bash
# Default: auto-discover latest merged CSVs
python plot_dse.py --dataset_name gscdv2

# Custom output directory
python plot_dse.py --dataset_name gscdv2 --output_dir ./plot/dse

# Specify exact CSV files
python plot_dse.py --retrain_csv ./log/gscdv2/retrain/merged/merged_test_20260206.csv \
                   --pretrain_csv ./log/gscdv2/pretrain/merged/merged_test_20260206.csv

# Skip bit sparsity analysis (faster if no saved models)
python plot_dse.py --dataset_name gscdv2 --no_bit_sparsity
```

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `gscdv2` | Dataset to plot |
| `--retrain_csv` | auto | Path to merged retrain CSV (auto-discovers latest) |
| `--pretrain_csv` | auto | Path to merged pretrain CSV (auto-discovers latest) |
| `--output_dir` | `./plot/dse` | Output directory for plots |
| `--log_dir` | `./log` | Root log directory for auto-discovery |
| `--source` | `test` | Which merged source to use: `best` or `test` |
| `--no_bit_sparsity` | off | Skip bit-level sparsity analysis from saved models |

### Plot Categories

`plot_dse.py` generates 10+ plots organized into 7 categories:

1. **Accuracy Landscape**: Gamma×threshold heatmap; accuracy vs gamma lines per threshold
2. **Sparsity Decomposition**: 3-panel heatmaps (weight, temporal, combined); grouped bar breakdown; temporal component lines (Δx, Δh, Δxh)
3. **Speedup Analysis**: 2-panel heatmap (speedup × and FLOPs reduction %); grouped bars for all 5 speedup scenarios
4. **Accuracy-Speedup Tradeoff**: Scatter plot with Pareto frontier
5. **Bit-Level Sparsity**: Two's complement vs sign-magnitude comparison; with/without MSQ comparison
6. **Summary Dashboard**: 2×3 multi-panel overview
7. **Pretrain Comparison**: LSTM baseline vs DeltaLSTM accuracy and accuracy drop

All plots are saved as both PNG (300dpi) and PDF.

### Output Directory Structure

After a DSE run, outputs are organized as:

```
./log/gscdv2/
├── pretrain/
│   ├── best/          # Per-run CSV/JSON/XLSX (validation-best epoch)
│   ├── test/          # Per-run CSV/JSON/XLSX (test results)
│   ├── hist/          # Per-run full training history
│   └── merged/        # Consolidated CSV/JSON/XLSX from merge_logs.py
│       ├── merged_best_20260206_143021.csv
│       ├── merged_best_20260206_143021.json
│       ├── merged_best_20260206_143021.xlsx
│       ├── merged_test_20260206_143022.csv
│       └── ...
├── retrain/
│   ├── best/
│   ├── test/
│   ├── hist/
│   └── merged/
./plot/dse/
├── accuracy_heatmap.png / .pdf
├── accuracy_vs_gamma.png / .pdf
├── sparsity_heatmaps.png / .pdf
├── sparsity_breakdown_bars.png / .pdf
├── temporal_sparsity_lines.png / .pdf
├── speedup_heatmap.png / .pdf
├── speedup_scenarios_bar.png / .pdf
├── accuracy_vs_speedup.png / .pdf
├── summary_dashboard.png / .pdf
└── pretrain_vs_retrain.png / .pdf
./save/gscdv2/
├── pretrain/          # Saved .pt model checkpoints (LSTM)
└── retrain/           # Saved .pt model checkpoints (DeltaLSTM)
```

### Parsing Merged CSVs

```python
import pandas as pd

# Load merged retrain test results
df = pd.read_csv('./log/gscdv2/retrain/merged/merged_test_20260206_143025.csv')

# Hyperparameters are in HP_* columns (parsed from model ID)
print(df[['MODEL_ID', 'HP_SPRNN', 'HP_TX', 'HP_TH', 'TEST_ACCURACY']])

# Filter: high accuracy configs with temporal sparsity
good = df[(df['TEST_ACCURACY'] > 0.90) & (df['HP_TX'] > 0)]
print(good[['HP_SPRNN', 'HP_TX', 'TEST_ACCURACY', 'TEST_SP_DXH']])
```

## References

- **DeltaRNN**: Gao et al., "DeltaRNN: A Power-efficient Recurrent Neural Network Accelerator", FPGA 2018
- **CBTD**: Gao et al., "Column-Balanced Targeted Dropout for RNN Sparsification"
- **MSQ**: "Memory-Efficient Bit Sparsification Quantization", ICCV 2025 - Sinusoidal LSB regularization to zero lower bits of quantized weights (`utils/msq_regularizer.py`)
- **BitWave**: "BitWave: Exploiting Column-Based Bit-Level Sparsity for Deep Learning Acceleration", HPCA 2025 - Sign-magnitude encoding for improved bit-level sparsity in negative weights (`utils/sparsity_analysis.py`)
