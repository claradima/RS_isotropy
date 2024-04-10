import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
import pandas as pd
import os
import json

from Geometry import *
def get_all_pmt_positions():

    # no inputs
    # output: 3xN numpy array, positions of all PMTs in Cartesian

    # reads csv file with PMT information
    # rescales info so that all position vectors have norm 1

    # note: csv file created with scratch/GetPMTPositions.cc, which
    # includes info for how to compile and run

    # load the CSV file with pmt x, y z coordinates
    data = np.genfromtxt('pmt_positions.csv', delimiter=',', skip_header=1)

    # rescale to vectors of length 1
    vector_lengths = np.linalg.norm(data, axis=1)
    rescaled_data = data / vector_lengths[:, np.newaxis]

    return rescaled_data

def plot_pmt_positions():

    # no inputs
    # no outpus

    # plots PMT map in phi-theta plane

    pmt_positions = get_all_pmt_positions()
    pmt_positions_polars = convert_points_to_spherical(pmt_positions)

    points_phi = np.zeros(len(pmt_positions_polars))
    points_theta = np.zeros(len(pmt_positions_polars))

    for i in range(len(pmt_positions_polars)):
        points_phi[i] = pmt_positions_polars[i][1]
        points_theta[i] = pmt_positions_polars[i][2]

    # plot PMTs on 2D map
    plt.scatter(points_phi, points_theta, s=0.5, color='blue')  # Adjust marker size and color as needed

    # set x-axis ticks
    plt.xticks(np.linspace(-np.pi, np.pi, 5),
               ['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

    # set y-axis ticks
    plt.yticks(np.linspace(0, np.pi, 3),
               ['$0$', '$\pi/2$', '$\pi$'])

    plt.xlabel('Phi')
    plt.ylabel('Theta')
    plt.title('All PMTs')

    # set aspect ratio to 'equal'
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    # create a folder named "plots" if it doesn't exist in one directory back
    plots_dir = os.path.join('..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # save the plot in the "plots" folder one directory back
    plt.savefig(os.path.join(plots_dir, 'PMTMap.pdf'), format='pdf')
    #plt.show()

def get_custom_pmt_positions_from_file(pmt_file_name):

    # input: string pmt_file_name
    # output: 3xN numpy array, positions of all PMTs in Cartesian

    # reads csv file with PMT information
    # rescales info so that all position vectors have norm 1
    # plots pmt map and saves it to ../plots

    # note: csv file created with scratch/GetPMTPositions.cc, which
    # includes info for how to compile and run

    # Define the file name
    json_file = 'PMT_files.json'
    # check valid pmt_file_name
    with open(json_file, 'r') as f:
        PMT_files = json.load(f)

    if pmt_file_name in PMT_files.values():
        # load the CSV file with pmt x, y z coordinates
        data = np.genfromtxt(pmt_file_name, delimiter=',', skip_header=1)

        # rescale to vectors of length 1
        vector_lengths = np.linalg.norm(data, axis=1)
        rescaled_data = data / vector_lengths[:, np.newaxis]
    else:
        raise FileNotFoundError('PMT file not found: {}'.format(pmt_file_name))

    return rescaled_data

def plot_custom_pmt_positions(pmt_file_name):
    # input: pmt file name (including .csv)
    # no outputs

    # plots PMT map in phi-theta plane
    # save in ../plots folder

    pmt_positions = get_custom_pmt_positions_from_file(pmt_file_name)
    pmt_positions_polars = convert_points_to_spherical(pmt_positions)

    points_phi = np.zeros(len(pmt_positions_polars))
    points_theta = np.zeros(len(pmt_positions_polars))

    for i in range(len(pmt_positions_polars)):
        points_phi[i] = pmt_positions_polars[i][1]
        points_theta[i] = pmt_positions_polars[i][2]

    # plot PMTs on 2D map
    plt.scatter(points_phi, points_theta, s=0.5, color='blue')  # Adjust marker size and color as needed

    # set x-axis ticks
    plt.xticks(np.linspace(-np.pi, np.pi, 5),
               ['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

    # set y-axis ticks
    plt.yticks(np.linspace(0, np.pi, 3),
               ['$0$', '$\pi/2$', '$\pi$'])

    plt.xlabel('Phi')
    plt.ylabel('Theta')
    plt.title('All PMTs')

    # set aspect ratio to 'equal'
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    # create a folder named "plots" if it doesn't exist in one directory back
    plots_dir = os.path.join('..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # save the plot in the "plots" folder one directory back
    plt.savefig(os.path.join(plots_dir, 'CustomPMTMap.pdf'), format='pdf')
    # plt.show()

def make_pmt_files_dict():

    # no inputs
    # no outputs
    # make JSON file with predefined PMT lists

    PMT_files = {'All PMTs': 'pmt_positions.csv'}

    # define the file name
    json_file = 'PMT_files.json'

    # save the dictionary to a JSON file
    with open(json_file, 'w') as f:
        json.dump(PMT_files, f)

def add_pmt_file_to_dict(element_name, file_name):

    # inputs: element_name = str, name of new list, include quotes
            # file_name = name of associated .csv file, include .csv in name
    # no outputs

    # modifies PMT_files.json
    # add new name of predefined PMT list to JSON file

    # load dictionary from the JSON file
    json_file = 'PMT_files.json'
    with open(json_file, 'r') as f:
        PMT_files = json.load(f)

    # add another element to the dictionary
    PMT_files[element_name] = file_name

    # save the updated dictionary back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(PMT_files, f)

def make_pmt_file_with_hole():

    # no inputs
    # no outputs

    # makes new .csv file - list of PMT map with hole

    # note: if you also want to plot this, use plot_custom_pmt_positions

    # load existing PMT files dictionary
    json_file = 'PMT_files.json'
    with open(json_file, 'r') as f:
        PMT_files = json.load(f)

    # make new list of PMTs
    all_pmt_positions = get_all_pmt_positions()
    pmt_positions = all_pmt_positions[400:]

    # save pmt_positions as a CSV file with the title "gappy_pmts.csv"
    with open('gappy_pmts.csv', 'w') as f:
        for pmt in pmt_positions:
            f.write(','.join(map(str, pmt)) + '\n')

    # spdate PMT_files dictionary with the new key-value pair
    PMT_files['PMTs with big gap'] = 'gappy_pmts.csv'

    # save the updated PMT_files dictionary back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(PMT_files, f)


def make_custom_pmt_file(mask, key_name, csv_file_name):

    # inputs: mask (list with true/false values of length len(all_pmt_positions)
            # key_name: name for new pmt list; use quotation marks
            # csv_file_name: name for associated csv file; include .csv
    # outputs: no outputs

    # Load existing PMT files dictionary
    json_file = 'PMT_files.json'
    with open(json_file, 'r') as f:
        PMT_files = json.load(f)

    # get all PMT positions and select PMTs based on the mask
    all_pmt_positions = get_all_pmt_positions()
    pmt_positions = all_pmt_positions[mask]

    # save pmt_positions as a CSV file with the provided file name
    with open(csv_file_name, 'w') as f:
        for pmt in pmt_positions:
            f.write(','.join(map(str, pmt)) + '\n')

    # update PMT_files dictionary with the new key-value pair
    PMT_files[key_name] = csv_file_name

    # save the updated PMT_files dictionary back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(PMT_files, f)




