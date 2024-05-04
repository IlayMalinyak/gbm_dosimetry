# __author:IlayK
# data:16/04/2024

import matplotlib.pyplot as plt
from utils import *

def plot_seeds_and_dose(ax, dose_coords, seeds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = plot_seeds(fixed_seeds, ax, color='r')
    ax.scatter(in_dose_coords[0], in_dose_coords[1], in_dose_coords[2], marker='o', alpha=0.1)
