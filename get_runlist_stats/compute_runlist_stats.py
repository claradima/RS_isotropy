#!/usr/bin/env python
# this assumes you already ran full_loop_optimized.py for your desired runlist and all the results are ready

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute coverage and isotropy stats for given runlist"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="nickel_coverage_stats.csv",
        help="Path to the input CSV (default: random_coverage_stats.csv)"
    )
    args = parser.parse_args()
    return args.input_csv


def make_unique_dir_from_csv(input_csv: str, parent_dir: str) -> str:
    """
    Given an input CSV filename and a parent directory, derive a base name
    from everything before '_coverage_stats.csv' (or else the stem), then
    append '_new' as needed until a non-existent directory name is found.
    Creates and returns the full path of that new directory.
    """
    os.makedirs(parent_dir, exist_ok=True)

    base = os.path.basename(input_csv)
    suffix = "_coverage_stats.csv"
    if base.endswith(suffix):
        name = base[:-len(suffix)]
    else:
        name = os.path.splitext(base)[0]

    candidate = name
    full_path = os.path.join(parent_dir, candidate)
    while os.path.exists(full_path):
        candidate += "_new"
        full_path = os.path.join(parent_dir, candidate)

    os.makedirs(full_path, exist_ok=False)
    return full_path


if __name__ == "__main__":
    csv_file = parse_args()
    results_parent_dir = (
        "/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/runlist_stats"
    )

    out_dir = make_unique_dir_from_csv(csv_file, results_parent_dir)
    runlist_name = os.path.basename(out_dir)
    print(f"Writing results into: {out_dir}")

    # Read the CSV
    df = pd.read_csv(csv_file)

    # Filter out rows where run_number is not numeric
    df = df[pd.to_numeric(df['run_number'], errors='coerce').notnull()].copy()
    df['run_number'] = df['run_number'].astype(int)

    # Identify metric columns (everything except 'run_number')
    metrics = [col for col in df.columns if col != 'run_number']

    # Process each metric
    for metric in metrics:
        values = df[metric].dropna().values
        n_entries = len(values)
        mean = np.mean(values)
        stdev = np.std(values, ddof=1)
        norm_stdev = stdev / mean if mean != 0 else float('nan')

        # Plot histogram
        plt.figure()
        plt.hist(values, bins='auto', alpha=0.7)
        plt.title(f"{runlist_name}_{metric}_stats")
        plt.xlabel(metric)
        plt.ylabel("Count")

        # Shade region mean ± stdev
        plt.axvspan(mean - stdev, mean + stdev, alpha=0.2)

        # Draw mean line
        plt.axvline(mean, color='k', linestyle='--', label=f"mean = {mean:.3f}")

        # Legend with entries, mean, stdev, normalized stdev
        legend_text = [
            f"entries: {n_entries}",
            f"mean = {mean:.3f}",
            f"stdev = {stdev:.3f}",
            f"norm stdev = {norm_stdev:.3f}"
        ]
        plt.legend(legend_text)

        # Save as PDF
        out_path = os.path.join(out_dir, f"{runlist_name}_{metric}_stats.pdf")
        plt.savefig(out_path, format='pdf')
        plt.close()

    print("All metrics processed and plots saved.")

