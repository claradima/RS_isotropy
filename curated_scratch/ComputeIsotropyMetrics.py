import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
import pandas as pd
from scipy.stats import skew, kurtosis



### Script computes the following for a given PMT configuration :
###
###     1) Vectorial sum of PMT positions (distances from center scaled to 1) - x, y, z coords + norm
###     2) Cap count distributions for alpha = pi/10, pi/8, pi/6, pi/4, pi/3 with:
###         i) Total N, mean, median
###         ii) Var, stdev, normalized stdev
###         iii) Maybe other stuff, add at the end (better ways to characterize distribution with left tail?)
###
### The nodes used to compute cap count distributions are fixed and stored in More_Nodes_Grid.csv 
### !!!!! Nodes positions in normalized spherical polars

### The PMT information is currently stored as follows:
###     pmt_positions.csv - contains the Cartesian coordinates of ALL PMTs
###     removed_indices - folder that contains one file for each custom PMT config; would like to turn these into
###                       files with a list of INCLUDED indices, not REMOVED (TO DO! Make correct files)


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

# Function to save indices to .txt file

def save_indices_to_file(indices, pmt_set_num):
    filename = f"random_indices_PMTSet{pmt_set_num}.txt"
    np.savetxt(filename, indices, fmt='%d')
    print(f"Saved random indices for PMTSet {pmt_set_num} to {filename}\n")

def bowley_skewness(data):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)
    return (q3 - 2 * q2 + q1) / (q3 - q1)


def moors_kurtosis(data):
    p12_5 = np.percentile(data, 12.5)
    p37_5 = np.percentile(data, 37.5)
    p62_5 = np.percentile(data, 62.5)
    p87_5 = np.percentile(data, 87.5)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)

    return ((p87_5 - p12_5) + (p62_5 - p37_5)) / (p75 - p25)


def get_active_positions(indices_file, all_pos_cart, all_pos_sph):
    indices = np.loadtxt(indices_file, dtype=int)
    print(indices.shape)
    print(all_pos_cart.shape)
    print(all_pos_sph.shape)
    active_positions_cart = all_pos_cart[indices]
    active_positions_sph = all_pos_sph[indices]
    return active_positions_cart, active_positions_sph

def compute_config_stats(active_pos_cart, active_pos_sph, nodes_pos_cart, nodes_pos_sph, set_name):
    # Create a new plot of all PMTs and Nodes
    plt.scatter(active_pos_sph[:, 1], active_pos_sph[:, 2], s=0.5, color='blue')  # Original points
    plt.scatter(nodes_pos_sph[:, 1], nodes_pos_sph[:, 2], s=3, color='red', label='Cap Nodes')  # Clicked points in red

    plt.xlabel('Phi')
    plt.ylabel('Theta')
    plt.title('PMTs and Nodes')

    vector_sum = np.sum(active_pos_cart, axis=0)
    vector_sum_length = np.linalg.norm(vector_sum)

    # Add a text box with the value of vectorial_sum and its length
    # text = f'Vectorial Sum: {vector_sum}\nLength: {vector_length:.2f}'
    # plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, va='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()

    plt.savefig(f'{set_name}_map.pdf', format='pdf')
    plt.show()

    pi_over_values = [3, 4, 6, 8, 10]

    # Create or initialize the CSV file
    # We will use a DataFrame to store stats
    stats_df = pd.DataFrame()
    stats_df['Metric'] = ['vector sum x','vector sum y','vector sum z','vector sum norm','total N', 'mean', 'median', 'variance', 'stdev', 'normalized stdev', 'skewness', 'kurtosis', 'bowley skewness', 'moors kurtosis']

    print("Now looping over alpha values")

    for pi_over_index in range(len(pi_over_values)):

        # Initialize points_in_cap_count outside the loop over pi_over values
        points_in_cap_count = np.zeros(len(nodes_pos_cart), dtype=int)

        pi_over = pi_over_values[pi_over_index]

        print("for alpha = pi / " + str(pi_over))

        print("Computing cap counts for alpha = pi / " + str(pi_over))

        for i in range(len(nodes_pos_cart)):
            cap_center = nodes_pos_cart[i]
            cap_center_polars = nodes_pos_sph[i]
            points_in_cap_set = []

            for j in range(len(active_pos_cart)):
                angle = angle_between_vectors(cap_center, active_pos_cart[j])
                if angle < np.pi / pi_over:
                    points_in_cap_set.append(active_pos_cart[j])

            points_in_cap_set = np.array(points_in_cap_set)

            points_in_cap_count[i] += len(points_in_cap_set)  # Accumulate count within each iteration

        print("Cap counts computed; now generating histogram and calculating stats ... ")

        N = len(active_pos_cart)  # Ensure you use the correct length for N here

        peak = peak_value(N, np.pi / pi_over)

        # Compute histogram
        hist_values, bin_edges = np.histogram(points_in_cap_count, bins=np.arange(points_in_cap_count.max() + 2))

        # Plot histogram
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

        plt.axvline(x=peak, color='red', linestyle='--', label='Expected Peak: {}'.format(peak))

        # Compute mean, median, and variance, skew, kurtosis

        alpha = f'alpha = pi/{pi_over}'
        # Compute the relevant statistics
        total_N = N
        mean_value = np.mean(points_in_cap_count)
        median_value = np.median(points_in_cap_count)
        variance_value = np.var(points_in_cap_count)
        stdev_value = np.sqrt(variance_value)
        normalized_stdev = stdev_value / mean_value
        skewness_value = skew(points_in_cap_count)
        kurtosis_value = kurtosis(points_in_cap_count)
        bowley_skewness_value = bowley_skewness(points_in_cap_count)
        moors_kurtosis_value = moors_kurtosis(points_in_cap_count)

        # Plot mean, median, and variance as vertical lines
        plt.axvline(x=mean_value, color='green', linestyle='--', label='Mean: {:.2f}'.format(mean_value))
        plt.axvline(x=median_value, color='purple', linestyle='--', label='Median: {:.2f}'.format(median_value))

        # Add variance and normalized standard deviation to legend

        # Add text for skewness and kurtosis
        plt.text(0.05, 0.45,
                 f'Skewness: {skewness_value:.2f}\nBSkewness: {bowley_skewness_value:.2f}\nKurtosis: {kurtosis_value:.2f}\nMKurtosis: {moors_kurtosis_value:.2f}\nNormalized stdev: {normalized_stdev:.6f}',
                 horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5), fontsize=font_size)

        # Add legend with larger font size
        plt.legend(fontsize=font_size)
        plt.title("Stats, alpha = pi / " + str(pi_over), fontsize=font_size)

        # Save the plot with a larger font size
        plt.savefig(f'Stats_{set_name}_PiOver{pi_over}.pdf', format='pdf')
        plt.show()

        # Add the statistics to the DataFrame
        stats_df[alpha] = [
            vector_sum[0],
            vector_sum[1],
            vector_sum[2],
            vector_sum_length,
            total_N,
            mean_value,
            median_value,
            variance_value,
            stdev_value,
            normalized_stdev,
            skewness_value,
            kurtosis_value,
            bowley_skewness_value,
            moors_kurtosis_value
        ]

        # %%
        # Save the DataFrame to CSV
        output_path = f'{set_name}_stats.csv'
        print(f"Saving CSV to: {output_path}")
        print(stats_df)
        stats_df.to_csv(output_path, index=False)
        print("CSV saved successfully.")


# Import PMT positions and normalize

all_pos_cart = np.genfromtxt('pmt_positions.csv', delimiter=',', skip_header=1)
all_pos_cart_rescaled = all_pos_cart / np.linalg.norm(all_pos_cart, axis=1)[:, np.newaxis]
all_pos_sph = convert_points_to_spherical(all_pos_cart_rescaled)

# Import nodes positions

nodes_pos_sph = np.genfromtxt('More_Nodes_Grid.csv', delimiter=',', skip_header=1)
nodes_pos_cart = convert_points_to_cartesian(nodes_pos_sph)

set_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

for i in set_numbers:
    print(f"Now working on PMT set {i}")
    indices_file = f'PMTSet{i}_indices.txt'
    active_pos_cart, active_pos_sph = get_active_positions(indices_file, all_pos_cart, all_pos_sph)
    compute_config_stats(active_pos_cart, active_pos_sph, nodes_pos_cart, nodes_pos_sph, f"PMTSet{i}")
