import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
import pandas as pd
import os
import json
import sys

from Geometry import *
from PMTs import *

response = input('Hello there, are you ready to make some plots? Type yes or no :) ')

if response.lower() == 'yes':
    print('Great! Let\'s make some plots.')
    # Call your plot-making function or do whatever you want
elif response.lower() == 'no':
    print('No problem. Maybe next time!')
    sys.exit()  # Exit the script if the user answers "no"
else:
    print('Hey man, that\'s kinda rude, please type something valid. I\'m just a computer, I don\'t understand random bits of human language. I\'ll assume you said yes so we eill move on ;)')

response = input('Type \'ok!\' to plot all the PMTs. You\'ll find this in the plots folder')

if response.lower() == 'ok!':
    plot_pmt_positions()
    print('Have a look in the folder')
else:
    print('Invalid again? This makes me sad. Come back when you want to play nicely')
    sys.exit()

