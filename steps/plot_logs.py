"""
Generate plots from existing training log files.

Usage:
    python steps/plot_logs.py --logfile log/gscdv2/pretrain/hist/MODEL_ID.csv
    python steps/plot_logs.py --log_dir log/gscdv2/pretrain/hist/  # Process all logs in directory
    python steps/plot_logs.py --dataset_name gscdv2 --phase pretrain  # Process all logs for dataset/phase
"""
import os
import argparse
from glob import glob
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_plotter import TrainingPlotter, create_plotter_from_log


def plot_single_log(logfile, plot_dir='./plot'):
    """Generate plots from a single log file."""
    print(f"Processing: {logfile}")
    try:
        plotter = create_plotter_from_log(logfile, plot_dir)
        plotter.save_history_csv()
        print(f"  -> Plots saved to: {plotter.plot_dir}")
    except Exception as e:
        print(f"  -> Error: {e}")


def plot_directory(log_dir, plot_dir='./plot'):
    """Generate plots from all log files in a directory."""
    log_files = glob(os.path.join(log_dir, '*.csv'))
    print(f"Found {len(log_files)} log files in {log_dir}")

    for logfile in sorted(log_files):
        plot_single_log(logfile, plot_dir)


def plot_dataset_phase(dataset_name, phase, plot_dir='./plot'):
    """Generate plots for all logs of a specific dataset and phase."""
    log_dir = os.path.join('./log', dataset_name, phase, 'hist')
    if os.path.exists(log_dir):
        plot_directory(log_dir, plot_dir)
    else:
        print(f"Log directory not found: {log_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate training plots from log files')
    parser.add_argument('--logfile', type=str, default=None,
                        help='Path to a single log CSV file')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Path to directory containing log CSV files')
    parser.add_argument('--dataset_name', type=str, default=None,
                        choices=['timit', 'gscdv2'],
                        help='Dataset name to process')
    parser.add_argument('--phase', type=str, default=None,
                        choices=['pretrain', 'retrain'],
                        help='Training phase to process')
    parser.add_argument('--plot_dir', type=str, default='./plot',
                        help='Base directory for output plots')
    parser.add_argument('--all', action='store_true',
                        help='Process all available logs')

    args = parser.parse_args()

    if args.logfile:
        plot_single_log(args.logfile, args.plot_dir)
    elif args.log_dir:
        plot_directory(args.log_dir, args.plot_dir)
    elif args.dataset_name and args.phase:
        plot_dataset_phase(args.dataset_name, args.phase, args.plot_dir)
    elif args.all:
        # Process all datasets and phases
        for dataset in ['timit', 'gscdv2']:
            for phase in ['pretrain', 'retrain']:
                print(f"\n=== {dataset.upper()} - {phase} ===")
                plot_dataset_phase(dataset, phase, args.plot_dir)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python steps/plot_logs.py --logfile log/gscdv2/pretrain/hist/MODEL.csv")
        print("  python steps/plot_logs.py --dataset_name gscdv2 --phase pretrain")
        print("  python steps/plot_logs.py --all")


if __name__ == '__main__':
    main()
