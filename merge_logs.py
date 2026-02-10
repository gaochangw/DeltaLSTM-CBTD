"""
merge_logs.py — Consolidated Log Merger for Design Space Scans

Merges best-case or test results from all runs into unified CSV, JSON, and XLSX
files for LLM parsing and cross-run analysis.

Usage:
    python merge_logs.py --dataset_name gscdv2 --phase retrain --source test
    python merge_logs.py --dataset_name gscdv2 --phase retrain --source best
    python merge_logs.py --dataset_name timit --phase pretrain --source best --sort_by VAL_PER
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model ID Parsing
# ---------------------------------------------------------------------------

def parse_model_id(model_id: str) -> dict:
    """Parse model_id into a dict of hyperparameter key-value pairs.

    The model_id format is strictly alternating KEY_VALUE pairs joined by '_'.
    Values may contain dots (floats), '+' (scientific notation), or text.
    """
    parts = model_id.split('_')
    params = {}
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        # Accumulate value parts: keep joining with '_' if the next part
        # doesn't look like a known key and isn't purely uppercase alpha.
        val_parts = [parts[i + 1]]
        j = i + 2
        # Scientific notation values like '0.0e+00' get split into ['0.0e+00']
        # but the '+' survives inside a single part. Edge case: value might be
        # a multi-word string (e.g. 'DeltaLSTM'). We greedily consume until
        # the next part is a known key at an even-pair boundary.
        while j < len(parts):
            # Check if parts[j] looks like a key (all uppercase alpha)
            if re.match(r'^[A-Z]+$', parts[j]) and j + 1 < len(parts):
                break
            val_parts.append(parts[j])
            j += 1
        value_str = '_'.join(val_parts)
        params[key] = _coerce_value(value_str)
        i = j
    return params


def _coerce_value(s: str):
    """Try to convert a string to int, float, or leave as str."""
    # Try int first (no dots, no 'e')
    if re.match(r'^-?\d+$', s):
        return int(s)
    # Try float (including scientific notation like 0.0e+00)
    try:
        return float(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Run Discovery
# ---------------------------------------------------------------------------

def discover_runs(log_root: str, dataset: str, phase: str, source: str
                  ) -> List[Dict[str, Any]]:
    """Discover all runs in the given log directory.

    Returns a list of dicts with keys:
        model_id, csv_path, json_path (or None), xlsx_path (or None)
    """
    source_dir = os.path.join(log_root, dataset, phase, source)
    if not os.path.isdir(source_dir):
        print(f"Error: source directory not found: {source_dir}")
        sys.exit(1)

    # 1. Find all CSVs
    csv_files = sorted(f for f in os.listdir(source_dir) if f.endswith('.csv'))
    if not csv_files:
        print(f"No CSV files found in {source_dir}")
        sys.exit(1)

    runs = []
    for csv_file in csv_files:
        model_id = csv_file[:-4]  # strip .csv
        runs.append({
            'model_id': model_id,
            'csv_path': os.path.join(source_dir, csv_file),
            'json_path': None,
            'xlsx_path': None,
        })

    # 2. Build model_id -> json_path map from JSON files in source dir
    json_map = _build_json_map(source_dir)

    # Also check hist/ directory as a fallback for training-phase JSONs
    hist_dir = os.path.join(log_root, dataset, phase, 'hist')
    if os.path.isdir(hist_dir) and hist_dir != source_dir:
        hist_json_map = _build_json_map(hist_dir)
        # hist is lower priority — don't overwrite source matches
        for mid, jpath in hist_json_map.items():
            if mid not in json_map:
                json_map[mid] = jpath

    # 3. Match JSON/XLSX to runs
    for run in runs:
        mid = run['model_id']
        if mid in json_map:
            run['json_path'] = json_map[mid]
            # Derive XLSX path from JSON
            xlsx_path = json_map[mid].replace('.json', '.xlsx')
            if os.path.isfile(xlsx_path):
                run['xlsx_path'] = xlsx_path

    return runs


def _build_json_map(directory: str) -> Dict[str, str]:
    """Scan directory for JSON files and map model_id -> json_path."""
    json_map = {}
    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(directory, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            mid = data.get('metadata', {}).get('model_id', '')
            if mid:
                # If multiple JSONs match the same model_id, keep the latest
                if mid not in json_map or fpath > json_map[mid]:
                    json_map[mid] = fpath
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return json_map


# ---------------------------------------------------------------------------
# CSV Merge
# ---------------------------------------------------------------------------

# Columns that contain bloated per-class dicts / ROC data
_BLOAT_PATTERNS = ('PER_CLASS', 'ROC_DATA', 'FRR_PER_CLASS')


def merge_csvs(runs: List[dict], keep_per_class: bool = False,
               sort_by: Optional[str] = None) -> pd.DataFrame:
    """Merge single-row CSVs from all runs into one DataFrame."""
    frames = []
    for run in runs:
        try:
            df = pd.read_csv(run['csv_path'])
        except Exception as e:
            print(f"  Warning: could not read {run['csv_path']}: {e}")
            continue

        # Add MODEL_ID column
        df.insert(0, 'MODEL_ID', run['model_id'])

        # Add parsed hyperparameter columns
        params = parse_model_id(run['model_id'])
        for k, v in params.items():
            df[f'HP_{k}'] = v

        frames.append(df)

    if not frames:
        print("Error: no CSVs could be loaded.")
        sys.exit(1)

    merged = pd.concat(frames, join='outer', ignore_index=True)

    # Drop bloated per-class columns by default
    if not keep_per_class:
        drop_cols = [c for c in merged.columns
                     if any(p in c for p in _BLOAT_PATTERNS)]
        if drop_cols:
            merged.drop(columns=drop_cols, inplace=True)

    # Sort if requested
    if sort_by and sort_by in merged.columns:
        ascending = any(kw in sort_by for kw in ('LOSS', 'PER', 'ERROR', 'EER', 'FAR'))
        merged.sort_values(sort_by, ascending=ascending, inplace=True)
        merged.reset_index(drop=True, inplace=True)

    return merged


# ---------------------------------------------------------------------------
# JSON Merge
# ---------------------------------------------------------------------------

def merge_jsons(runs: List[dict], dataset: str, phase: str, source: str
                ) -> Optional[dict]:
    """Merge JSON logs from all runs (skipping training_history)."""
    run_entries = []
    for run in runs:
        if not run['json_path']:
            continue
        try:
            with open(run['json_path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read {run['json_path']}: {e}")
            continue

        entry = {
            'model_id': run['model_id'],
            'json_source': os.path.basename(run['json_path']),
            'config': data.get('config', {}),
            'complexity': data.get('complexity', {}),
            'best_results': data.get('best_results', {}),
            'final_sparsity': data.get('final_sparsity', {}),
            'parsed_params': parse_model_id(run['model_id']),
        }
        run_entries.append(entry)

    if not run_entries:
        return None

    merged = {
        'merge_metadata': {
            'created_at': datetime.now().isoformat(),
            'dataset': dataset,
            'phase': phase,
            'source': source,
            'n_runs': len(run_entries),
        },
        'runs': run_entries,
    }
    return merged


# ---------------------------------------------------------------------------
# XLSX Summary
# ---------------------------------------------------------------------------

# ExcelLogger-compatible colors
_COLORS = {
    'header_bg': '4472C4',
    'header_fg': 'FFFFFF',
    'subheader_bg': '8EA9DB',
}

# Key metric columns to include (order preserved)
_KEY_METRICS = [
    'VAL_ACCURACY', 'TEST_ACCURACY',
    'VAL_F1_MACRO', 'TEST_F1_MACRO',
    'VAL_PER', 'TEST_PER',
    'VAL_ERROR_RATE', 'TEST_ERROR_RATE',
    'VAL_EER_MACRO', 'TEST_EER_MACRO',
    'VAL_AUC_MACRO', 'TEST_AUC_MACRO',
    'SP_W_RNN', 'SP_W_FC',
    'VAL_SP_DX', 'VAL_SP_DH', 'VAL_SP_DXH',
    'TEST_SP_DX', 'TEST_SP_DH', 'TEST_SP_DXH',
    'SP_BIT_RNN', 'SP_BIT_NZ_RNN',
    'SP_BIT_SM_RNN', 'SP_BIT_NZ_SM_RNN',
    'MSQ_LAMBDA', 'MSQ_K', 'BITWAVE',
    'N_PARAM',
]

# Columns where values are ratios (0-1) that should be shown as percentages
_PCT_COLUMNS = {
    'VAL_ACCURACY', 'TEST_ACCURACY',
    'VAL_F1_MACRO', 'TEST_F1_MACRO',
    'VAL_PER', 'TEST_PER',
    'VAL_ERROR_RATE', 'TEST_ERROR_RATE',
    'VAL_EER_MACRO', 'TEST_EER_MACRO',
    'VAL_AUC_MACRO', 'TEST_AUC_MACRO',
    'SP_W_RNN', 'SP_W_FC',
    'VAL_SP_DX', 'VAL_SP_DH', 'VAL_SP_DXH',
    'TEST_SP_DX', 'TEST_SP_DH', 'TEST_SP_DXH',
    'SP_BIT_RNN', 'SP_BIT_NZ_RNN',
    'SP_BIT_SM_RNN', 'SP_BIT_NZ_SM_RNN',
}


def merge_xlsx_summary(merged_df: pd.DataFrame, output_path: str):
    """Create a formatted single-sheet XLSX summary of all runs."""
    if not OPENPYXL_AVAILABLE:
        print("  Skipping XLSX: openpyxl not installed.")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "All Runs"

    # Build column list: MODEL_ID + HP_* columns + available key metrics
    hp_cols = sorted(c for c in merged_df.columns if c.startswith('HP_'))
    metric_cols = [c for c in _KEY_METRICS if c in merged_df.columns]
    columns = ['MODEL_ID'] + hp_cols + metric_cols

    # Header styling
    header_font = Font(bold=True, color=_COLORS['header_fg'], size=11)
    header_fill = PatternFill(start_color=_COLORS['header_bg'],
                              end_color=_COLORS['header_bg'],
                              fill_type='solid')
    header_align = Alignment(horizontal='center', vertical='center')
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    data_align = Alignment(horizontal='center', vertical='center')

    # Write headers (row 1)
    for col_idx, col_name in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border

    # Write data rows
    for row_idx, (_, row_data) in enumerate(merged_df.iterrows(), 2):
        for col_idx, col_name in enumerate(columns, 1):
            val = row_data.get(col_name)
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.alignment = data_align
            cell.border = thin_border

            if pd.isna(val):
                cell.value = '-'
            elif col_name in _PCT_COLUMNS and isinstance(val, (int, float)):
                cell.value = f"{val * 100:.2f}%"
            elif isinstance(val, float):
                if abs(val) < 0.01 and val != 0:
                    cell.value = f"{val:.2e}"
                else:
                    cell.value = round(val, 4)
            else:
                cell.value = val

    # Auto-adjust column widths
    for column_cells in ws.columns:
        max_length = 0
        col_letter = get_column_letter(column_cells[0].column)
        for cell in column_cells:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_length + 3, 30)

    wb.save(output_path)


# ---------------------------------------------------------------------------
# Console Summary
# ---------------------------------------------------------------------------

def print_summary(runs: List[dict], merged_df: pd.DataFrame,
                  output_paths: Dict[str, str]):
    """Print a merge summary to the console."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        _rich_summary(console, runs, merged_df, output_paths)
    except ImportError:
        _plain_summary(runs, merged_df, output_paths)


def _rich_summary(console, runs, merged_df, output_paths):
    from rich.table import Table

    console.print()
    console.rule("[bold blue]Merge Summary[/bold blue]")

    table = Table(show_header=False, show_edge=False, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Total runs", str(len(runs)))
    n_json = sum(1 for r in runs if r['json_path'])
    n_xlsx = sum(1 for r in runs if r['xlsx_path'])
    table.add_row("Matched JSON", f"{n_json}/{len(runs)}")
    table.add_row("Matched XLSX", f"{n_xlsx}/{len(runs)}")
    table.add_row("Merged columns", str(len(merged_df.columns)))
    console.print(table)

    # Metric ranges
    metric_cols = [c for c in _KEY_METRICS if c in merged_df.columns]
    if metric_cols:
        console.print()
        mt = Table(title="Metric Ranges", show_lines=True)
        mt.add_column("Metric", style="bold")
        mt.add_column("Min", justify="right")
        mt.add_column("Max", justify="right")
        mt.add_column("Mean", justify="right")

        for col in metric_cols:
            vals = pd.to_numeric(merged_df[col], errors='coerce').dropna()
            if vals.empty:
                continue
            is_pct = col in _PCT_COLUMNS
            fmt = (lambda v: f"{v * 100:.2f}%") if is_pct else (lambda v: f"{v:.4f}")
            mt.add_row(col, fmt(vals.min()), fmt(vals.max()), fmt(vals.mean()))

        console.print(mt)

    # Output files
    console.print()
    for label, path in output_paths.items():
        console.print(f"  [green]{label}[/green]: {path}")
    console.print()


def _plain_summary(runs, merged_df, output_paths):
    print()
    print("=" * 60)
    print("Merge Summary")
    print("=" * 60)
    print(f"  Total runs:     {len(runs)}")
    n_json = sum(1 for r in runs if r['json_path'])
    n_xlsx = sum(1 for r in runs if r['xlsx_path'])
    print(f"  Matched JSON:   {n_json}/{len(runs)}")
    print(f"  Matched XLSX:   {n_xlsx}/{len(runs)}")
    print(f"  Merged columns: {len(merged_df.columns)}")

    metric_cols = [c for c in _KEY_METRICS if c in merged_df.columns]
    if metric_cols:
        print()
        print(f"  {'Metric':<25s} {'Min':>12s} {'Max':>12s} {'Mean':>12s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
        for col in metric_cols:
            vals = pd.to_numeric(merged_df[col], errors='coerce').dropna()
            if vals.empty:
                continue
            is_pct = col in _PCT_COLUMNS
            fmt = (lambda v: f"{v * 100:.2f}%") if is_pct else (lambda v: f"{v:.4f}")
            print(f"  {col:<25s} {fmt(vals.min()):>12s} {fmt(vals.max()):>12s} {fmt(vals.mean()):>12s}")

    print()
    for label, path in output_paths.items():
        print(f"  {label}: {path}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge log files from multiple runs into consolidated CSV/JSON/XLSX.")
    parser.add_argument('--dataset_name', required=True, choices=['timit', 'gscdv2'],
                        help='Dataset name')
    parser.add_argument('--phase', required=True, choices=['pretrain', 'retrain'],
                        help='Training phase')
    parser.add_argument('--source', default='best', choices=['best', 'test', 'hist'],
                        help='Which CSVs to merge (default: best)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: ./log/{dataset}/{phase}/merged/)')
    parser.add_argument('--keep_per_class', action='store_true',
                        help='Keep per-class dict columns (dropped by default)')
    parser.add_argument('--no_json', action='store_true',
                        help='Skip JSON merge')
    parser.add_argument('--no_xlsx', action='store_true',
                        help='Skip XLSX summary')
    parser.add_argument('--sort_by', default=None,
                        help='Column to sort merged results by (e.g. TEST_ACCURACY)')
    parser.add_argument('--log_dir', default='./log',
                        help='Root log directory (default: ./log)')

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        args.log_dir, args.dataset_name, args.phase, 'merged')
    os.makedirs(output_dir, exist_ok=True)

    # Discover runs
    print(f"Scanning: {args.log_dir}/{args.dataset_name}/{args.phase}/{args.source}/")
    runs = discover_runs(args.log_dir, args.dataset_name, args.phase, args.source)
    print(f"Found {len(runs)} run(s).")

    output_paths = {}

    # 1. CSV merge
    merged_df = merge_csvs(runs, keep_per_class=args.keep_per_class,
                           sort_by=args.sort_by)
    csv_out = os.path.join(output_dir, f"merged_{args.source}_{timestamp}.csv")
    merged_df.to_csv(csv_out, index=False)
    output_paths['CSV'] = csv_out

    # 2. JSON merge
    if not args.no_json:
        merged_json = merge_jsons(runs, args.dataset_name, args.phase, args.source)
        if merged_json:
            json_out = os.path.join(output_dir, f"merged_{args.source}_{timestamp}.json")
            with open(json_out, 'w', encoding='utf-8') as f:
                json.dump(merged_json, f, indent=2)
            output_paths['JSON'] = json_out
        else:
            print("  No JSON logs found to merge.")

    # 3. XLSX summary
    if not args.no_xlsx:
        xlsx_out = os.path.join(output_dir, f"merged_{args.source}_{timestamp}.xlsx")
        merge_xlsx_summary(merged_df, xlsx_out)
        if OPENPYXL_AVAILABLE:
            output_paths['XLSX'] = xlsx_out

    # Summary
    print_summary(runs, merged_df, output_paths)


if __name__ == '__main__':
    main()
