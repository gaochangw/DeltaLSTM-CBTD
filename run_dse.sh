#!/bin/bash
# ==============================================================================
# run_dse.sh — Design Space Exploration for GSCDv2 DeltaLSTM-CBTD
#
# Sweeps over weight sparsity (gamma_rnn), temporal sparsity (th_x/th_h),
# and bit sparsity (msq_lambda) to map the full accuracy–sparsity–speedup
# design space.
#
# Prerequisites (run once before this script):
#   python main.py --dataset_name gscdv2 --step data_prep --data_dir /path/to/gscdv2
#   python main.py --dataset_name gscdv2 --step feat_ext
#
# Usage:
#   bash run_dse.sh                            # Run exploration
#   bash run_dse.sh 2>&1 | tee dse_log.txt     # Save full console log
# ==============================================================================

set -o pipefail

# ==============================================================================
#  TUNABLE PARAMETERS — Edit these for your exploration
# ==============================================================================

DATASET="gscdv2"
GPU=0
SEED=4

# --- Training epochs ---
# NOTE: ALPHA_ANNEAL_EPOCH must be ≥ 2 (AAE=1 causes division by zero in train.py).
#       For meaningful CBTD, set PRETRAIN_EPOCHS ≥ AAE.
PRETRAIN_EPOCHS=2              # Full: 50
RETRAIN_EPOCHS=1               # Full: 5
ALPHA_ANNEAL_EPOCH=2           # Full: 30  (must be ≥ 2; set ≤ PRETRAIN_EPOCHS)

# --- Weight sparsity: CBTD target (gamma_rnn = gamma_fc) ---
GAMMAS="0.0 0.94"              # Full: "0.0 0.5 0.8 0.9 0.94"

# --- Temporal sparsity: Delta thresholds (th_x = th_h) ---
THRESHOLDS="0.0 0.25"          # Full: "0.0 0.125 0.25 0.5"

# --- Bit sparsity: MSQ regularization (pretrain only) ---
MSQ_LAMBDAS="0.0"              # Full: "0.0 1.0"
MSQ_K=2

# --- Directories ---
LOG_DIR="./log"
PLOT_DIR="./plot/dse"

# ==============================================================================
#  END TUNABLE PARAMETERS
# ==============================================================================

# Convert to arrays
read -ra GAMMA_ARR  <<< "$GAMMAS"
read -ra TH_ARR     <<< "$THRESHOLDS"
read -ra MSQ_ARR    <<< "$MSQ_LAMBDAS"

N_PRETRAIN=$(( ${#GAMMA_ARR[@]} * ${#MSQ_ARR[@]} ))
N_RETRAIN=$(( N_PRETRAIN * ${#TH_ARR[@]} ))
TOTAL=$(( N_PRETRAIN * 2 + N_RETRAIN * 2 ))  # train+test for each

echo "============================================================"
echo " Design Space Exploration — $DATASET"
echo "============================================================"
echo " Gammas:      ${GAMMAS}"
echo " Thresholds:  ${THRESHOLDS}"
echo " MSQ lambdas: ${MSQ_LAMBDAS}"
echo " Pretrain:    ${PRETRAIN_EPOCHS} epochs (AAE=${ALPHA_ANNEAL_EPOCH})"
echo " Retrain:     ${RETRAIN_EPOCHS} epochs"
echo " Grid:        ${N_PRETRAIN} pretrain × ${#TH_ARR[@]} thresholds = ${N_RETRAIN} configs"
echo " Total steps: ${TOTAL} (train + test per config)"
echo "============================================================"
echo ""

FAILED=0
STEP=0
START_TIME=$SECONDS

# ---------- helper ----------
run_step() {
    local desc="$1"; shift
    STEP=$((STEP + 1))
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo " [${STEP}/${TOTAL}] ${desc}"
    echo "────────────────────────────────────────────────────────────"
    if "$@"; then
        echo " ✓ Done"
    else
        echo " ✗ FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
}

# ==========================================================================
#  Phase 1 — Pretrain LSTM (sweep gamma × msq_lambda)
# ==========================================================================
echo ""
echo "========== Phase 1: Pretrain LSTM =========="

for gamma in "${GAMMA_ARR[@]}"; do
  for msq in "${MSQ_ARR[@]}"; do

    COMMON_ARGS=(
        --dataset_name "$DATASET" --gpu_device "$GPU" --seed "$SEED"
        --hid_type LSTM
        --gamma_rnn "$gamma" --gamma_fc "$gamma"
        --alpha_anneal_epoch "$ALPHA_ANNEAL_EPOCH"
    )
    # Only pass MSQ args when lambda > 0 to keep model IDs clean
    MSQ_ARGS=()
    if (( $(echo "$msq > 0" | bc -l) )); then
        MSQ_ARGS=(--msq_lambda "$msq" --msq_k "$MSQ_K")
    fi

    run_step "Pretrain LSTM  gamma=${gamma} msq=${msq}" \
        python main.py "${COMMON_ARGS[@]}" "${MSQ_ARGS[@]}" \
            --step train --phase pretrain --n_epochs "$PRETRAIN_EPOCHS"

    run_step "Test pretrain  gamma=${gamma} msq=${msq}" \
        python main.py "${COMMON_ARGS[@]}" "${MSQ_ARGS[@]}" \
            --step test --phase pretrain

  done
done

# ==========================================================================
#  Phase 2 — Retrain DeltaLSTM (sweep thresholds for each pretrained model)
# ==========================================================================
echo ""
echo "========== Phase 2: Retrain DeltaLSTM =========="

for gamma in "${GAMMA_ARR[@]}"; do
  for msq in "${MSQ_ARR[@]}"; do
    for th in "${TH_ARR[@]}"; do

      COMMON_ARGS=(
          --dataset_name "$DATASET" --gpu_device "$GPU" --seed "$SEED"
          --gamma_rnn "$gamma" --gamma_fc "$gamma"
          --th_x "$th" --th_h "$th"
          --alpha_anneal_epoch "$ALPHA_ANNEAL_EPOCH"
      )
      MSQ_ARGS=()
      if (( $(echo "$msq > 0" | bc -l) )); then
          MSQ_ARGS=(--msq_lambda "$msq" --msq_k "$MSQ_K")
      fi

      run_step "Retrain DeltaLSTM  gamma=${gamma} msq=${msq} th=${th}" \
          python main.py "${COMMON_ARGS[@]}" "${MSQ_ARGS[@]}" \
              --hid_type DeltaLSTM --hid_type_pretrain LSTM \
              --step train --phase retrain --n_epochs_retrain "$RETRAIN_EPOCHS"

      run_step "Test retrain       gamma=${gamma} msq=${msq} th=${th}" \
          python main.py "${COMMON_ARGS[@]}" "${MSQ_ARGS[@]}" \
              --hid_type DeltaLSTM \
              --step test --phase retrain

    done
  done
done

# ==========================================================================
#  Phase 3 — Merge all logs
# ==========================================================================
echo ""
echo "========== Phase 3: Merge Logs =========="

python merge_logs.py --dataset_name "$DATASET" --phase pretrain --source best \
    --sort_by VAL_ACCURACY
python merge_logs.py --dataset_name "$DATASET" --phase pretrain --source test \
    --sort_by TEST_ACCURACY
python merge_logs.py --dataset_name "$DATASET" --phase retrain --source best \
    --sort_by VAL_ACCURACY
python merge_logs.py --dataset_name "$DATASET" --phase retrain --source test \
    --sort_by TEST_ACCURACY

# ==========================================================================
#  Phase 4 — Generate plots
# ==========================================================================
echo ""
echo "========== Phase 4: Plot Results =========="

python plot_dse.py --dataset_name "$DATASET" --output_dir "$PLOT_DIR"

# ==========================================================================
#  Summary
# ==========================================================================
ELAPSED=$(( SECONDS - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo " DSE Complete — ${DATASET}"
echo "============================================================"
echo " Total time:   ${MINS}m ${SECS}s"
echo " Steps run:    ${STEP}"
echo " Failed:       ${FAILED}"
echo " Logs:         ${LOG_DIR}/${DATASET}/"
echo " Merged:       ${LOG_DIR}/${DATASET}/{pretrain,retrain}/merged/"
echo " Plots:        ${PLOT_DIR}/"
echo "============================================================"
