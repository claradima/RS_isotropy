import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process a run number.")
    parser.add_argument(
        "--run-number",
        type=int,
        default=300000,
        help="Run number (integer). Default is 300000 - but that's bad, deffo change it."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default='random_coverage_stats.csv',
        help="Name of output csv file with coverage and isotropy metrics. The stats for the run will be added to that file"
    )
    parser.add_argument(
        "--output-pmtmap-dir",
        type=str,
        default='random_list',
        help="Name of the directory where PMT map PDFs will be saved"
    )
    args = parser.parse_args()
    return args.run_number, args.output_csv, args.output_pmtmap_dir

### Script to compute coverage metrics for a given run

# define some functions

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return r, phi, theta

def convert_points_to_spherical(points):
    return np.array([cartesian_to_spherical(x, y, z) for x, y, z in points])

def spherical_to_cartesian(r, phi, theta):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z

def convert_points_to_cartesian(points):
    return np.array([spherical_to_cartesian(r, phi, theta) for r, phi, theta in points])

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

def peak_value(N, alpha):
    return N * (1 - np.cos(alpha)) / 2

# Functions to load active PMT positions

def get_active_positions_old(run_number):
    phi_theta_file = f'/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/phi_theta/{run_number}_phi_theta.csv'
    active_pos_sph = np.loadtxt(phi_theta_file, delimiter=",", dtype=float)
    active_pos_cart = convert_points_to_cartesian(active_pos_sph)
    return active_pos_cart, active_pos_sph

def get_active_positions(run_number):
    phi_theta_file = f'/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/phi_theta/{run_number}_phi_theta.csv'
    active_pos_sph = np.loadtxt(phi_theta_file, delimiter=",", dtype=float)
    original_count = active_pos_sph.shape[0]
    active_pos_sph = np.unique(active_pos_sph, axis=0)
    removed = original_count - active_pos_sph.shape[0]
    if removed > 0:
        print(f"⚠️ Removed {removed} duplicate entries from {run_number}_phi_theta.csv")
    active_pos_cart = convert_points_to_cartesian(active_pos_sph)
    return active_pos_cart, active_pos_sph

def plot_pmt_map(run_number, pmtmap_dir):
    set_name = f'{run_number}_PMTMap'
    plt.scatter(active_pos_sph[:, 1], active_pos_sph[:, 2], s=0.5, color='blue')
    plt.scatter(nodes_pos_sph[:, 1], nodes_pos_sph[:, 2], s=3, color='red', label='Cap Nodes')
    plt.xlabel('Phi')
    plt.ylabel('Theta')
    plt.title(f'PMTs map for run {run_number}')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(
        '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/PMT_maps',
        pmtmap_dir,
        f'{set_name}.pdf'
    )
    plt.savefig(out_path, format='pdf')
    plt.close()

def create_results_csv(filename):
    headers = [
        "run_number", "total_N", "vector_sum_norm",
        "pio3_stdev", "pio4_stdev", "pio6_stdev", "pio8_stdev",
        "pio3_stdev_n", "pio4_stdev_n", "pio6_stdev_n", "pio8_stdev_n"
    ]
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"CSV file '{filename}' created with headers only.")

def compute_coverage_stats(alpha, run_number, active_pos_cart, active_pos_sph, nodes_pos_cart, nodes_pos_sph):
    counts = []
    print(f"Computing cap counts for alpha = {alpha}")
    for cap_center in nodes_pos_cart:
        angles = np.arccos(
            np.clip(
                np.dot(active_pos_cart, cap_center) /
                (np.linalg.norm(active_pos_cart, axis=1) * np.linalg.norm(cap_center)),
                -1.0, 1.0
            )
        )
        counts.append(np.count_nonzero(angles < alpha))
    counts = np.array(counts)
    mean_value = counts.mean()
    stdev_value = counts.std()
    normalized_stdev = stdev_value / mean_value
    return stdev_value, normalized_stdev

def append_coverage_stats(
    filename, run_number, total_N, vector_sum_norm,
    pio3_stdev, pio4_stdev, pio6_stdev, pio8_stdev,
    pio3_stdev_n, pio4_stdev_n, pio6_stdev_n, pio8_stdev_n
):
    row = [
        run_number, total_N, vector_sum_norm,
        pio3_stdev, pio4_stdev, pio6_stdev, pio8_stdev,
        pio3_stdev_n, pio4_stdev_n, pio6_stdev_n, pio8_stdev_n
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    print(f"Added row for run {run_number} to '{filename}'.")

if __name__ == "__main__":
    run_number, output_csv_filename, output_pmtmap_dir = parse_args()
    print(f"run number: {run_number}\n")
    print('Getting nodes coordinates')
    nodes_pos_sph = np.genfromtxt('More_Nodes_Grid.csv', delimiter=',', skip_header=1)
    nodes_pos_cart = convert_points_to_cartesian(nodes_pos_sph)
    print('Nodes coordinates loaded\n')
    active_pos_cart, active_pos_sph = get_active_positions(run_number)
    print('Extracted coordinates of PMTs')

    # Ensure output directory exists
    out_dir = os.path.join(
        '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/PMT_maps',
        output_pmtmap_dir
    )
    os.makedirs(out_dir, exist_ok=True)
    print('Generating PMT map...')
    plot_pmt_map(run_number, output_pmtmap_dir)
    print('PMT map saved!\n')

    # Check/c**reate stats CSV
    if not os.path.isfile(output_csv_filename):
        create_results_csv(output_csv_filename)
    else:
        print(f"CSV File '{output_csv_filename}' already exists.")

    print(f'Computing alpha-independent stats for run {run_number}')
    total_N = len(active_pos_cart)
    vector_sum_norm = np.linalg.norm(active_pos_cart.sum(axis=0))
    print(f'Alpha-independent stats done for run {run_number}\n')

    # Alpha-dependent stats
    stats = {}
    for name, alpha in [('pio3', np.pi/3), ('pio4', np.pi/4), ('pio6', np.pi/6), ('pio8', np.pi/8)]:
        print(f'Computing for alpha = {alpha} for run {run_number}')
        stdev, stdev_n = compute_coverage_stats(
            alpha, run_number,
            active_pos_cart, active_pos_sph,
            nodes_pos_cart, nodes_pos_sph
        )
        stats[name] = (stdev, stdev_n)
        print(f'alpha = {alpha} done for run {run_number}\n')

    print(f'Appending stats to csv file for run {run_number}')
    append_coverage_stats(
        output_csv_filename,
        run_number,
        total_N,
        vector_sum_norm,
        stats['pio3'][0], stats['pio4'][0], stats['pio6'][0], stats['pio8'][0],
        stats['pio3'][1], stats['pio4'][1], stats['pio6'][1], stats['pio8'][1]
    )
    print('All done :)')

