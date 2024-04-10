#############################################################################
### This file contains definitions of all the functions used in any of the
### Run Selection Isotropy scripts, as well as explanations for how they
### work. Enjoy!
###
### If you have been unfortunate enough to have to read any of this and you
### have questions, you can contact me at c.dima@sussex.ac.uk
###
### Have fun!
#############################################################################

import numpy as np


def cartesian_to_spherical(x, y, z):

    # inputs: x, y, z (floats)
    # outputs: r, phi, theta (floats)
    # converts cartesian coordinates to spherical
    # note: phi is the angle in the xy plane and theta is the polar angle

    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return r, phi, theta

def spherical_to_cartesian(r, phi, theta):

    # inputs: r, phi, theta (floats)
    # outputs: x, y, z (floats)

    # converts spherical polar coordinates to cartesian
    # note: phi is the angle in the xy plane and theta is the polar angle

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z

def convert_points_to_cartesian(points):

    # input: Nx3 numpy array 'points' with positions in spherical polars (unpacked below)
    # output: Nx3 numpy array 'points_set_cartesians' with positions in cartesian coords

    # applies spherical_to_cartesian function defined above for each point
    # appends transformed point to list
    # this appending is probably quite inefficient, might need to change

    points_set_cartesians = []
    for point in points:
        r = point[0]
        phi = point[1]
        theta = point[2]
        x, y, z = spherical_to_cartesian(r, phi, theta)
        points_set_cartesians.append((x, y, z))
    return np.array(points_set_cartesians)

def convert_points_to_spherical(points):

    # output: Nx3 numpy array 'points_set_polars' with positions in cartesian coords
    # input: Nx3 numpy array 'points' with positions in spherical polars (unpacked below)

    # applies caartesian_to_spherical function defined above for each point
    # appends transformed point to list
    # this appending is probably quite inefficient, might need to change

    points_set_polars = []
    for point in points:
        x = point[0]
        y = point[1]
        z = point[2]
        r, phi, theta = cartesian_to_spherical(x, y, z)
        points_set_polars.append((r, phi, theta))
    return np.array(points_set_polars)

def angle_between_vectors(v1, v2):

    # inputs: v1 = numpy array of length 3, cartesian coordinates of a vector
            # v2 = numpy array of length 3, cartesian coordinates of a vector
    # output: float angle_rad = angle between vectors in radians

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle_rad

