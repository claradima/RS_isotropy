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
        "-run_number",
        type=int,
        default=364600,
        help="Run number (integer). Default is 300000 - but that's bad, deffo change it."
    )
    args = parser.parse_args()
    return args.run_number

### Script to compute coverage metrics for a given run

# define some functions

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return r, phi, theta

def convert_points_to_spherical(points):
    points_set_polars = []
    for point in points:
        x = point[0]
        y = point[1]
        z = point[2]
        r, phi, theta = cartesian_to_spherical(x, y, z)
        points_set_polars.append((r, phi, theta))
    return np.array(points_set_polars)

def spherical_to_cartesian(r, phi, theta):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z

def convert_points_to_cartesian(points):
    points_set_cartesians = []
    for point in points:
        r = point[0]
        phi = point[1]
        theta = point[2]
        x, y, z = spherical_to_cartesian(r, phi, theta)
        points_set_cartesians.append((x, y, z))
    return np.array(points_set_cartesians)

#function that computes the angle between two vectors

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle_rad


def peak_value(N, alpha):
    return N * (1 - np.cos(alpha)) / 2

# Note: function below reads PMT coords form file that contains normalized
#       spherical coords; also returns cartesian coords

def get_active_positions(run_number):
    phi_theta_file = f'/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/phi_theta/{run_number}_phi_theta.csv'
    active_pos_sph = np.loadtxt(phi_theta_file, delimiter=",", dtype=float)
    #print(type(active_pos_sph))
    active_pos_cart = convert_points_to_cartesian(active_pos_sph)
    return active_pos_cart, active_pos_sph
    
def plot_pmt_map(run_number):
    set_name = f'{run_number}_PMTMap'
    # Create a new plot of all PMTs
    plt.scatter(active_pos_sph[:, 1], active_pos_sph[:, 2], s=0.5, color='blue')  
    plt.scatter(nodes_pos_sph[:, 1], nodes_pos_sph[:, 2], s=3, color='red', label='Cap Nodes')  # Clicked points in red

    plt.xlabel('Phi')
    plt.ylabel('Theta')
    plt.title(f'PMTs map for run {run_number}')

    vector_sum = np.sum(active_pos_cart, axis=0)
    vector_sum_length = np.linalg.norm(vector_sum)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()

    plt.savefig(f'/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/PMT_maps/nickel_list/{set_name}.pdf', format='pdf')
    #plt.show()
    
def create_results_csv(filename):
    headers = [
        "run_number",
        "total_N",
        "vector_sum_norm",
        "pio3_stdev",
        "pio4_stdev",
        "pio6_stdev",
        "pio8_stdev",
        "pio10_stdev",
        "pio3_stdev_n",
        "pio4_stdev_n",
        "pio6_stdev_n",
        "pio8_stdev_n",
        "pio10_stdev_n"
    ]
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"CSV file '{filename}' created with headers only.")
    
def compute_coverage_stats(alpha, run_number, active_pos_cart, active_pos_sph, nodes_pos_cart, nodes_pos_sph):
    
    points_in_cap_count = np.zeros(len(nodes_pos_cart), dtype=int)
    print(f"Computing cap counts for alpha = {alpha}")
    
    for i in range(len(nodes_pos_cart)):
            cap_center = nodes_pos_cart[i]
            cap_center_polars = nodes_pos_sph[i]
            points_in_cap_set = []

            for j in range(len(active_pos_cart)):
                angle = angle_between_vectors(cap_center, active_pos_cart[j])
                if angle < alpha:
                    points_in_cap_set.append(active_pos_cart[j])

            points_in_cap_set = np.array(points_in_cap_set)

            points_in_cap_count[i] += len(points_in_cap_set)  # Accumulate count within each iteration

    # Compute coverage metrics

    print("Cap counts computed; now calculating stats ... ")

    #total_N = len(active_pos_cart)  # Ensure you use the correct length for N here
    mean_value = np.mean(points_in_cap_count)
    variance_value = np.var(points_in_cap_count)
    stdev_value = np.sqrt(variance_value)
    normalized_stdev = stdev_value / mean_value
        
    #vector_sum = np.sum(active_pos_cart, axis=0)
    #vector_sum_norm = np.linalg.norm(vector_sum)
        
        
        
        
        
        
        
    '''
    # Plot histogram
    hist_values, bin_edges = np.histogram(points_in_cap_count, bins=np.arange(points_in_cap_count.max() + 2))
    
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist_values, width=0.5, align='center')

    # Set larger font sizes
    font_size = 16
    plt.xlabel('Number of points in cap', fontsize=font_size)
    plt.ylabel('Number of directions', fontsize=font_size)
    plt.title('Histogram of points in cap per direction', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Find the first non-zero bin edge
    non_zero_index = np.nonzero(hist_values)[0][0]

    # Automatically adjust x-axis limits
    xmin = bin_edges[non_zero_index] - 20
    xmax = max(bin_edges) + 20
    plt.xlim(xmin, xmax)

    plt.grid(True)

    # Plot mean, median, and variance as vertical lines
    plt.axvline(x=mean_value, color='green', linestyle='--', label='Mean: {:.2f}'.format(mean_value))

    # Add legend with larger font size
    plt.legend(fontsize=font_size)
    plt.title("Stats, alpha = " + str(alpha), fontsize=font_size)

    # Save the plot with a larger font size
    plt.savefig(f'Stats_{run_number}_Alpha{alpha}.pdf', format='pdf')
    #plt.show()
        
    '''
        
        
        
        
        
    return stdev_value, normalized_stdev
    
def append_coverage_stats(
    filename,
    run_number,
    total_N,
    vector_sum_norm,
    pio3_stdev,
    pio4_stdev,
    pio6_stdev,
    pio8_stdev,
    pio3_stdev_n,
    pio4_stdev_n,
    pio6_stdev_n,
    pio8_stdev_n
):
    row = [
        run_number,
        total_N,
        vector_sum_norm,
        pio3_stdev,
        pio4_stdev,
        pio6_stdev,
        pio8_stdev,
        pio3_stdev_n,
        pio4_stdev_n,
        pio6_stdev_n,
        pio8_stdev_n
    ]
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    print(f"Added row for run {run_number} to '{filename}'.")

if __name__ == "__main__":

    # Get run number from argument parser
    
    run_number = parse_args()
    print(f"run number: {run_number}")
    print(' ')
    print('Getting nodes coordinates')
    
    # Import nodes positions

    nodes_pos_sph = np.genfromtxt('More_Nodes_Grid.csv', delimiter=',', skip_header=1)
    nodes_pos_cart = convert_points_to_cartesian(nodes_pos_sph)
    print('Nodes coordinates loaded')
    print('  ')
    
    #Extract PMT coordinates
    
    active_pos_cart, active_pos_sph = get_active_positions(run_number)
    print('Extracted coordinates of PMTs')
    print('Generating PMT map...')
    
    # Plot PMT map
    
    plot_pmt_map(run_number)
    
    print('PMT map saved!')
    print('  ')
    
    # Check if stats csv file exists; create it if not
    
    filename = "nickel_coverage_stats.csv"

    if not os.path.isfile(filename):
        create_results_csv(filename)
    else:
        print(f"CSV File '{filename}' already exists.")
        
    # Compute stats for run
    print('Computing alpha-independent stats')
    total_N = len(active_pos_cart)
    vector_sum = np.sum(active_pos_cart, axis=0)
    vector_sum_norm = np.linalg.norm(vector_sum)
    print('Alpha-independent stats done')
    print('  ')

    print('Computing alpha-dependent stats')
    print('Computing for alpha = pi/3')    
    pio3_stdev, pio3_stdev_n = compute_coverage_stats(alpha = (np.pi/3), run_number = run_number, active_pos_cart = active_pos_cart, active_pos_sph = active_pos_sph, nodes_pos_cart = nodes_pos_cart, nodes_pos_sph = nodes_pos_sph)
    print('alpha = pi/3 done')
    print('  ')
    
    print('Computing alpha-dependent stats')
    print('Computing for alpha = pi/4')    
    pio4_stdev, pio4_stdev_n = compute_coverage_stats(alpha = (np.pi/4), run_number = run_number, active_pos_cart = active_pos_cart, active_pos_sph = active_pos_sph, nodes_pos_cart = nodes_pos_cart, nodes_pos_sph = nodes_pos_sph)
    print('alpha = pi/4 done')
    print('  ')
    
    print('Computing alpha-dependent stats')
    print('Computing for alpha = pi/6')    
    pio6_stdev, pio6_stdev_n = compute_coverage_stats(alpha = (np.pi/6), run_number = run_number, active_pos_cart = active_pos_cart, active_pos_sph = active_pos_sph, nodes_pos_cart = nodes_pos_cart, nodes_pos_sph = nodes_pos_sph)
    print('alpha = pi/6 done')
    print('  ')
    
    print('Computing alpha-dependent stats')
    print('Computing for alpha = pi/8')    
    pio8_stdev, pio8_stdev_n = compute_coverage_stats(alpha = (np.pi/8), run_number = run_number, active_pos_cart = active_pos_cart, active_pos_sph = active_pos_sph, nodes_pos_cart = nodes_pos_cart, nodes_pos_sph = nodes_pos_sph)
    print('alpha = pi/8 done')
    print('  ')
    
    print('Computing alpha-dependent stats')
    print('Computing for alpha = pi/10')    
    pio10_stdev, pio10_stdev_n = compute_coverage_stats(alpha = (np.pi/10), run_number = run_number, active_pos_cart = active_pos_cart, active_pos_sph = active_pos_sph, nodes_pos_cart = nodes_pos_cart, nodes_pos_sph = nodes_pos_sph)
    print('alpha = pi/10 done')
    print('  ')
    
    # Add stats to csv file
    print(f'Appending stats to csv file for run {run_number}')
    append_coverage_stats(
    "nickel_coverage_stats.csv",
    run_number,
    total_N,
    vector_sum_norm,
    pio3_stdev,
    pio4_stdev,
    pio6_stdev,
    pio8_stdev,
    pio3_stdev_n,
    pio4_stdev_n,
    pio6_stdev_n,
    pio8_stdev_n
)
    print('All done :)')

