#!/usr/bin/env bash
set -euo pipefail

# Download the JaneEye-Net dataset from Kaggle competition files and
# normalize to the expected layout:
#   <dest>/event_data/train/...
#   <dest>/event_data/test/...

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${1:-${REPO_ROOT}/.datasets/janeeye}"
KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-${REPO_ROOT}/.kaggle}"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
VENV_KAGGLE="${REPO_ROOT}/.venv/bin/kaggle"
COMPETITION="event-based-eye-tracking-cvpr-2025"

if [[ ! -x "${VENV_KAGGLE}" ]]; then
  echo "Missing Kaggle CLI in ${REPO_ROOT}/.venv."
  echo "Install with: .venv/bin/pip install kaggle"
  exit 1
fi

mkdir -p "${DEST_DIR}" "${KAGGLE_CONFIG_DIR}"

# Auth modes:
# 1) KAGGLE_API_TOKEN (new Kaggle auth flow)
# 2) kaggle.json with username/key (legacy flow)
HAVE_TOKEN=0
if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
  HAVE_TOKEN=1
fi

if [[ ${HAVE_TOKEN} -eq 0 && ! -f "${KAGGLE_CONFIG_DIR}/kaggle.json" ]]; then
  cat <<EOF
Missing Kaggle auth.

Provide either:
  - env var: KAGGLE_API_TOKEN
or:
  - file: ${KAGGLE_CONFIG_DIR}/kaggle.json

1) Go to https://www.kaggle.com/settings/account
2) Create a new API token
3) Either export KAGGLE_API_TOKEN or save kaggle.json to:
     ${KAGGLE_CONFIG_DIR}/kaggle.json
4) Re-run this script.
EOF
  exit 2
fi

if [[ -f "${KAGGLE_CONFIG_DIR}/kaggle.json" ]]; then
  chmod 600 "${KAGGLE_CONFIG_DIR}/kaggle.json"
fi

RAW_DIR="${DEST_DIR}/_raw_kaggle"
mkdir -p "${RAW_DIR}"

echo "[1/3] Downloading Kaggle competition files..."
KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR}" \
  "${VENV_KAGGLE}" competitions download -c "${COMPETITION}" -p "${RAW_DIR}"

echo "[2/3] Extracting archives..."
shopt -s nullglob
for zf in "${RAW_DIR}"/*.zip; do
  echo "Extracting $(basename "${zf}")"
  "${VENV_PY}" -m zipfile -e "${zf}" "${DEST_DIR}"
done
shopt -u nullglob

echo "[3/3] Normalizing layout..."
if [[ -d "${DEST_DIR}/event_data" ]]; then
  :
elif [[ -d "${DEST_DIR}/3ET+ dataset/event_data" ]]; then
  mv "${DEST_DIR}/3ET+ dataset/event_data" "${DEST_DIR}/event_data"
  rm -rf "${DEST_DIR}/3ET+ dataset"
elif [[ -d "${DEST_DIR}/train" && -d "${DEST_DIR}/test" ]]; then
  mkdir -p "${DEST_DIR}/event_data"
  mv "${DEST_DIR}/train" "${DEST_DIR}/event_data/train"
  mv "${DEST_DIR}/test" "${DEST_DIR}/event_data/test"
else
  echo "Could not find expected extracted folders."
  echo "Inspect ${DEST_DIR} and adjust layout manually."
  exit 3
fi

echo "Done."
echo "JaneEye-Net data dir: ${DEST_DIR}/event_data"
