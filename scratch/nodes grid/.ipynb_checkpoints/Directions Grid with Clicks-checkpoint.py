import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
import pandas as pd

# Load the CSV file with pmt x, y z coordinates
data = np.genfromtxt('pmt_positions.csv', delimiter=',', skip_header=1)

# rescale to vectors of length 1

vector_lengths = np.linalg.norm(data, axis=1)
rescaled_data = data / vector_lengths[:, np.newaxis]
vector_lengths_rescaled = np.linalg.norm(rescaled_data, axis = 1)

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

# transform to polars to plot

points_polars = convert_points_to_spherical(rescaled_data)

# separate coordinates to plot more easily

#%%
points_phi = np.zeros(len(points_polars))
points_theta = np.zeros(len(points_polars))

for i in range(len(points_polars)):
    points_phi[i] = points_polars[i][1]
    points_theta[i] = points_polars[i][2]
# make clickable plot so you can find coordinates of points

#Plot points on 2D map

# Plot the grid points
plt.scatter(points_phi, points_theta, s=10, color='blue')  # Adjust marker size and color as needed


# Set x-axis ticks
plt.xticks(np.linspace(-np.pi, np.pi, 5),
           ['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

# Set y-axis ticks
plt.yticks(np.linspace(0, np.pi, 3),
           ['$0$', '$\pi/2$', '$\pi$'])

plt.xlabel('Phi')
plt.ylabel('Theta')
plt.title('All PMTs')

# Set aspect ratio to 'equal'
plt.gca().set_aspect('equal', adjustable='box')


plt.grid(True)

# Interactive click to get coordinates
clicked_points = plt.ginput(n=100, timeout=0)

print("Clicked points:", clicked_points)

#plt.savefig('PMTMap.pdf', format='pdf')
plt.show()

# Unpack the clicked points
points_phi_2, points_theta_2 = zip(*clicked_points)

# Create a new plot including the clicked points
plt.scatter(points_phi, points_theta, s=5, color='blue')  # Original points
plt.scatter(points_phi_2, points_theta_2, s=10, color='red', label='Clicked Points')  # Clicked points in red

plt.xlabel('Phi')
plt.ylabel('Theta')
plt.title('All PMTs with Clicked Points')

plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend()

# Save the plot as a PDF file
plt.savefig('Minimal_Nodes_Grid_Directions.pdf', format='pdf')

# Save the clicked points as a CSV file
clicked_points_df = pd.DataFrame({'Phi': points_phi_2, 'Theta': points_theta_2})
clicked_points_df.to_csv('Minimal_Nodes_Grid.csv', index=False)

# Add a new column filled with 1s
clicked_points_df.insert(0, 'Extra_Column', 1)

plt.show()