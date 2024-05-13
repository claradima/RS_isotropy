#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
import pandas as pd

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
    return points_set_polars


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
    return points_set_cartesians

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

# Load the CSV file with pmt x, y z coordinates
data0 = np.genfromtxt('pmt_positions.csv', delimiter=',', skip_header=1)

data1 = data0
data2 = data0[400:]
data3 = data0[800:]
data4 = data0[1200:]
data5 = data0[1600:]

# make PMTset data5 by randomly selecting 400 indices from data0 and excluding them
random_indices = np.random.choice(len(data0), size=400, replace=False)
data6 = np.delete(data0, random_indices, axis=0)

# make PMTset data5 by randomly selecting 800 indices from data0 and excluding them
random_indices = np.random.choice(len(data0), size=800, replace=False)
data7 = np.delete(data0, random_indices, axis=0)

# make PMTset data5 by randomly selecting 1200 indices from data0 and excluding them
random_indices = np.random.choice(len(data0), size=1200, replace=False)
data8 = np.delete(data0, random_indices, axis=0)

# make PMTset data5 by randomly selecting 1600 indices from data0 and excluding them
random_indices = np.random.choice(len(data0), size=1600, replace=False)
data9 = np.delete(data0, random_indices, axis=0)

all_data = [data1, data2, data3, data4, data5, data6, data7, data8, data9]

# get grid nodes

nodes = np.genfromtxt('More_Nodes_Grid.csv', delimiter=',', skip_header=1)

# Unpack the clicked points
points_phi_2, points_theta_2 = zip(*nodes)

# Create an array filled with 1s for the first column
column_of_ones = np.ones((nodes.shape[0], 1))

# Stack the column of ones with the last two columns of the original array
new_nodes = np.hstack((column_of_ones, nodes[:, 0].reshape(-1, 1), nodes[:, 1].reshape(-1, 1)))

points_set_2 = convert_points_to_cartesian(new_nodes)

for k in range(len(all_data)):
    data = all_data[k]

    print("PMT set " + str(k+1) + " received")

    # rescale to vectors of length 1

    vector_lengths = np.linalg.norm(data, axis=1)
    rescaled_data = data / vector_lengths[:, np.newaxis]
    vector_lengths_rescaled = np.linalg.norm(rescaled_data, axis=1)

    # transform to polars to plot

    points_polars = convert_points_to_spherical(rescaled_data)

    # separate coordinates to plot more easily

    points_polars = np.array(points_polars)  # Convert to NumPy array for better performance

    points_phi = points_polars[:, 1]
    points_theta = points_polars[:, 2]

    print("This is what it looks like: ")

    # Create a new plot including the clicked points
    plt.scatter(points_phi, points_theta, s=0.5, color='blue')  # Original points
    plt.scatter(points_phi_2, points_theta_2, s=3, color='red', label='Big Nodes')  # Clicked points in red

    plt.xlabel('Phi')
    plt.ylabel('Theta')
    plt.title('PMTs and Nodes, PMT Set' + str(k+1))

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()

    # Save the plot as a PDF file
    plt.savefig('PMTsAndNodes_PMTSet' + str(k+1) + '.pdf', format='pdf')

    plt.show()

    # Initialize points_in_cap_count outside the loop over pi_over values
    #points_in_cap_count = np.zeros(len(points_set_2))

    pi_over_values = [3, 4, 6, 8, 10]

    print("Now looping over alpha values")

    for pi_over_index in range(len(pi_over_values)):

        # Initialize points_in_cap_count outside the loop over pi_over values
        points_in_cap_count = np.zeros(len(points_set_2))

        pi_over = pi_over_values[pi_over_index]

        print("for alpha = pi / " + str(pi_over))

        print("Computing cap counts for alpha = pi / " + str(pi_over))

        for i in range(len(points_set_2)):
            cap_center = points_set_2[i]
            cap_center_polars = new_nodes[i]
            points_in_cap_set = []

            for j in range(len(rescaled_data)):
                angle = angle_between_vectors(cap_center, rescaled_data[j])
                if angle < np.pi / pi_over:
                    points_in_cap_set.append(rescaled_data[j])

            points_in_cap_set = np.array(points_in_cap_set)

            points_in_cap_count[i] += len(points_in_cap_set)  # Accumulate count within each iteration

        print("Cap counts computed; now generating histogram and calculating stats ... ")

        N = len(points_phi)  # Ensure you use the correct length for N here

        peak = peak_value(N, np.pi / pi_over)

        # Compute histogram
        hist_values, bin_edges = np.histogram(points_in_cap_count, bins=np.arange(points_in_cap_count.max() + 2))

        # Plot histogram
        plt.bar(bin_edges[:-1], hist_values, width=0.5, align='center')

        plt.xlabel('Number of points in cap')

        # Find the first non-zero bin edge
        non_zero_index = np.nonzero(hist_values)[0][0]

        # Automatically adjust x-axis limits
        xmin = bin_edges[non_zero_index] - 20
        xmax = max(bin_edges) + 20
        plt.xlim(xmin, xmax)

        plt.ylabel('Number of directions')
        plt.title('Histogram of points in cap per direction')
        plt.grid(True)

        plt.axvline(x=peak, color='red', linestyle='--', label='Expected Peak: {}'.format(peak))

        # Compute mean, median, and variance
        mean_value = np.mean(points_in_cap_count)
        median_value = np.median(points_in_cap_count)
        variance_value = np.var(points_in_cap_count)
        sqrt_variance = variance_value ** 0.5
        normalized_stdev = sqrt_variance / mean_value
        # Plot mean, median, and variance as vertical lines
        plt.axvline(x=mean_value, color='green', linestyle='--', label='Mean: {:.2f}'.format(mean_value))
        plt.axvline(x=median_value, color='purple', linestyle='--', label='Median: {:.2f}'.format(median_value))

        # Add variance and normalized standard deviation to legend
        plt.text(0.05, 0.45, 'sqrtVariance: {:.6f}\nnormalizedStdev: {:.6f}'.format(sqrt_variance, normalized_stdev),
                 horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5))

        # Add legend
        plt.legend()
        plt.title("Stats, PMT Set " + str(k + 1) + ", alpha = pi / " + str(pi_over))
        plt.savefig("Stats_PMTSet  "+ str(k + 1) + "_PiOver" + str(pi_over) + ".pdf", format='pdf')
        plt.show()