# __author:IlayK
# data:15/04/2024
import os

import matplotlib.pyplot as plt
import numpy as np
import re
import shutil

import pandas as pd

from utils import *
from dose import *
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation



LSEED = 10  # mm


def get_dose_from_ctr(dir):
    struct = read_structure(dir, 1)
    return struct[0][1].T

def create_cases_dirs(directory):
    """
    Create a directory for each experiment number and move the corresponding files to the directory
    :param directory: directory containing the xlsx files
    """
    pattern = r'Fixed dp13578_0_to_62 (1).xlsx'
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an Excel file and matches the naming pattern
        if filename.endswith('.xlsx'):
            exp_num = filename.split('_')[0].split('dp')[-1]
            # Create a directory for the experiment number if it doesn't exist
            exp_dir = os.path.join(directory, exp_num)
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)

            # Move the file to the experiment directory
            src_path = os.path.join(directory, filename)
            dest_path = os.path.join(exp_dir, filename)
            shutil.move(src_path, dest_path)

def read_xlsx(folder_path):
    """
    Read the xlsx files with pages per seed point (start, middle, end)
    :param folder_path: folder consists files
    :return: dictionary with keys as the case dates and values as the data
    """
    data_dict = dict()
    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            # Extract <title>, <day>, and <day2> from the filename
            parts = filename.split('_')
            title = parts[0].split(' ')[0]
            day = parts[1]
            day2 = parts[3].split('.')[0]

            # Read the Excel file
            excel_path = os.path.join(folder_path, filename)
            sheets = pd.read_excel(excel_path, sheet_name=None, header=None, engine='openpyxl')

            # Create the DataFrame
            values_list = []
            for sheet_name, sheet_data in sheets.items():
                values_list.append(sheet_data.values[:,None,:])

            # Stack the DataFrames along axis 1
            values_stacked = np.concatenate(values_list, axis=1)
            values_stacked = np.transpose(values_stacked, (1, 2, 0))

            # Store the DataFrame in the dictionary
            key = f"{title}_{day}_{day2}"
            data_dict[key] = values_stacked
    return data_dict

def get_seed_from_csv(dir):
    """
    Read the seeds from the csv files
    :param dir: dir containing the csv files
    :return:
    """
    all_seeds = []
    fixed, moving = None, None
    for f in os.listdir(dir):
        if f.endswith('.csv'):
            print(f.lower())
            if 'fixed' in f.lower():
                fixed = pd.read_csv(dir + '/' + f).values.reshape(-1, 3, 3)
                fixed = np.transpose(fixed, (1,2, 0))
                print(fixed[0])
            if 'moving' in f.lower():
                moving = pd.read_csv(dir + '/' + f).values.reshape(-1, 3, 3)
                moving = np.transpose(moving, (1,2, 0))
    all_seeds.append((fixed, moving))
    return all_seeds


def get_dose_coords(seeds, dose_LUT, R_LUT, THETA_LUT,x, y, z, dose_threshold=20):
    """
    Calculate the dose coordinates for the seeds
    :param seeds:
    :param dose_LUT:
    :param R_LUT:
    :param THETA_LUT:
    :param x:
    :param z:
    :param dose_threshold:
    :return:
    """
    dose = np.zeros(x.shape)
    dose = calc_dose_multiple_seeds(seeds, dose_LUT, R_LUT, THETA_LUT,
                                          x, y, z, dose)
    dose_idx = np.where(dose > dose_threshold)
    dose_coords = np.vstack((x[dose_idx[0], dose_idx[1], dose_idx[2]].flatten(),
                                   y[dose_idx[0], dose_idx[1], dose_idx[2]].flatten(),
                                   z[dose_idx[0], dose_idx[1], dose_idx[2]].flatten()))
    return dose_coords, dose


def calc_dose_outliers_bf(doseXYZ,x,y,z, seeds, res):
    """

    :param doseXYZ: nxnxn
    :param x: nxnxn
    :param y: nxnxn
    :param z: nxnxn
    :param seeds: 3x3xm
    :return:
    """
    outliers = []
    outliers_freq = np.zeros(seeds.shape[-1])
    outliers_dist = np.zeros(seeds.shape[-1])
    low_dose = np.logical_and(doseXYZ < 20, doseXYZ > 1e-3)
    high_dose = doseXYZ > 20
    high_dose_cords = grid_to_coords(x, y, z, np.where(high_dose))
    num_seed_points = int(LSEED / res)
    for s in range(seeds.shape[-1]):
        seed = seeds[..., s]
        seed = np.linspace(seed[0], seed[-1], 20)
        flat_x, flat_y, flat_z = x.flatten(), y.flatten(), z.flatten()
        dist = np.sqrt((seed[:, 0][:, None] - flat_x[None]) ** 2 + (seed[:, 1][:, None] - flat_y[None]) ** 2 + (
                    seed[:, 2][:, None] - flat_z[None]) ** 2)
        min_dist = np.argmin(dist, axis=-1)
        seed_grid = np.zeros_like(doseXYZ)
        seed_grid[np.unravel_index(min_dist, doseXYZ.shape)] = 1
        seed_cords = np.where(seed_grid)
        outl_grid = np.logical_and(seed_grid, low_dose)
        outl_idx = np.where(outl_grid)
        if len(outl_idx[0]):
            outl_coords = grid_to_coords(x, y, z, outl_idx) # Nx3
            dist = np.sqrt((high_dose_cords[:,0][:, None] - outl_coords[:,0][None]) ** 2
                           + (high_dose_cords[:,1][:, None] - outl_coords[:,1][None]) ** 2 + (
                    high_dose_cords[:,2][:, None] - outl_coords[:,2][None]) ** 2)
            min_dist = np.min(dist, axis=0)
            if len(outl_idx[0]) > 2 and np.max(min_dist) > res:
                print(f"Found {len(outl_idx[0])} outliers pointa at max distance {np.max(min_dist)}")
                outliers.append(outl_coords)
                outliers_dist[s] = np.max(min_dist)
                outliers_freq[s] = len(outl_idx[0]) / len(seed_cords[0])
    return outliers, np.array(outliers_freq), np.array(outliers_dist), None


def increase_convex_hull_density(convex_hull_vertices, d=10):
    # Perform Delaunay triangulation
    # Get convex hull vertices

    # chull = delaunay.convex_hull
    # # Extract vertices of convex hull
    # convex_hull_vertices = dose_cords[chull]

    # Add intermediate points along edges of the convex hull
    new_points = np.zeros((0, 3))
    for i in range(len(convex_hull_vertices) - 1):
        edge = convex_hull_vertices[i:i+2]
        edge_points = np.linspace(edge[0], edge[1], d + 2)[1:-1]  # Exclude endpoints
        new_points = np.vstack((new_points, edge_points))
    edge = np.array([convex_hull_vertices[-1], convex_hull_vertices[0]])
    edge_points = np.linspace(edge[0], edge[1], d + 2)[1:-1]  # Exclude endpoints
    new_points = np.vstack((new_points, edge_points))

    return new_points
def calc_dose_outliers_convexhull(dose_cords, seeds, split_seeds=False):
    delaunay = Delaunay(dose_cords.T)
    convex_hull = delaunay.convex_hull
    convex_dose = np.concatenate((dose_cords[0][convex_hull[:, 0]][None], dose_cords[1][convex_hull[:, 1]][None],
                           dose_cords[2][convex_hull[:, 2]][None]), axis=0)
    convex_hull_big = increase_convex_hull_density(convex_dose.T, d=100)
    outliers = []
    outliers_freq = []
    outliers_dist = []
    for s in range(seeds.shape[-1]):
        seed = seeds[..., s]
        seed = np.linspace(seed[0], seed[-1], 100)
        delaunay_points = delaunay.find_simplex(seed)
        out = np.where(delaunay_points == -1)
        outliers.append(seed[out])
        outliers_freq.append(len(out[0]) / len(seed))
        seed_dists = []
        for point in seed[out]:
            seed_dists.append(np.min(np.linalg.norm(dose_cords.T - point, axis=1)))
        if len(seed_dists):
            outliers_dist.append(max(seed_dists))
    return {"outliers": outliers, "outliers_freq": np.array(outliers_freq), "outliers_dist": np.array(outliers_dist)}

def grid_to_coords(x, y, z, idx):
    return np.vstack((x[idx[0], idx[1], idx[2]].flatten(),
                        y[idx[0], idx[1], idx[2]].flatten(),
                        z[idx[0], idx[1], idx[2]].flatten())).T
def calc_dose_overlap(dose_coords_fixed, dose_coords_moving):
    delaunay = Delaunay(dose_coords_fixed.T)
    delaunay_points = delaunay.find_simplex(dose_coords_moving.T)
    overlap = np.where(delaunay_points != -1)
    max_coords = np.max(np.concatenate([dose_coords_fixed, dose_coords_moving], axis=1), axis=1)
    min_coords = np.min(np.concatenate([dose_coords_fixed, dose_coords_moving], axis=1), axis=1)
    return dose_coords_moving[:, overlap[0]], max_coords, min_coords

def calc_dose_iou(fixed_XYZ, moving_XYZ):
    fixed_dose = np.zeros(fixed_XYZ.shape)
    moving_dose = np.zeros(moving_XYZ.shape)
    fixed_dose[fixed_XYZ > 20] = 1
    moving_dose[moving_XYZ > 20] = 1
    intersection = np.sum(np.logical_and(fixed_dose, moving_dose))
    union = np.sum(np.logical_or(fixed_dose, moving_dose))
    return intersection / union

def calc_overlap_error(res, max_coords, min_coords):
    interp_error = max(res**2/4, 0.1)
    coords_diff = max_coords - min_coords
    max_error_p = np.max(coords_diff) / interp_error
    return max_error_p

def cluster_seeds(seeds, n_clusters=2):
    seeds_center = np.mean(seeds, axis=1)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(seeds_center.T)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Separate the points into two clusters based on their labels
    cluster1_indices = np.where(labels == 0)[0]
    cluster2_indices = np.where(labels == 1)[0]
    return seeds[..., cluster1_indices], seeds[..., cluster2_indices]

def calc_cluster_characteristics(seeds, res, seperate=True):
    if seperate:
        seeds1, seeds2 = cluster_seeds(seeds)
        ax = plot_seeds(seeds1, color='r')
        ax = plot_seeds(seeds2, ax=ax, color='b')
        plt.show()
        coords1 = np.max(np.mean(seeds1,axis=1), axis=1) - np.min(np.mean(seeds1,axis=1), axis=-1)
        coords2 = np.max(np.mean(seeds2,axis=1), axis=1) - np.min(np.mean(seeds2,axis=1), axis=-1)
        max_coords = np.max(np.concatenate([coords1, coords2]))
    else:
        max_coords = np.max(np.max(np.mean(seeds,axis=1), axis=1) - np.min(np.mean(seeds,axis=1), axis=1))
    error_p = max_coords / res
    return max_coords, error_p

def check_ones_neighbours(dose, i, j, k):
    neighbours = []
    if i > 0:
        neighbours.append(dose[i-1, j, k])
    if j > 0:
        neighbours.append(dose[i, j-1, k])
    if k > 0:
        neighbours.append(dose[i, j, k-1])
    if i < dose.shape[0] - 1:
        neighbours.append(dose[i+1, j, k])
    if j < dose.shape[1] - 1:
        neighbours.append(dose[i, j+1, k])
    if k < dose.shape[2] - 1:
        neighbours.append(dose[i, j, k+1])
    return 1 in neighbours

def create_errorbar_dose(dose_XYZ, error_size, res):
    binary_dose = np.zeros(dose_XYZ.shape)
    binary_dose[dose_XYZ > 20] = 1
    expansion_scale = int(error_size / res)  # Expand by units of resolution
    errorbar_dose = binary_dose.copy()
    for _ in range(expansion_scale):
        neighbors = np.array([np.array((i,j,k)) for i in range(len(dose_XYZ)) for j in range(len(dose_XYZ[0]))
                     for k in range(len(dose_XYZ[0][0])) if (errorbar_dose[i,j,k]==0 and check_ones_neighbours(errorbar_dose, i,j,k))])
        errorbar_dose[neighbors[:,0], neighbors[:,1], neighbors[:,2]] = 1
    print(f"Expanding errorbar dose by {len(neighbors)} cells")
    # errorbar_coords = np.array(np.where(errorbar_dose))
    return errorbar_dose, binary_dose

def pair_experiment(fixed_seeds, moving_seeds, pair_name, res=0.4, err=1):
    fixed_cm = np.mean(fixed_seeds[1, :, :], axis=-1)
    moving_cm = np.mean(moving_seeds[1, :, :], axis=-1)
    cm = (fixed_cm + moving_cm) / 2
    fixed_seeds = fixed_seeds - cm[None, :, None]
    moving_seeds = moving_seeds - cm[None, :, None]
    max_coords = max(np.max(fixed_seeds), np.max(moving_seeds))
    min_coords = min(np.min(fixed_seeds), np.min(moving_seeds))
    w = max(abs(max_coords), abs(min_coords))
    dose_LUT, R_LUT, THETA_LUT = generate_dose_tables(DOSE_MAT_PATH2, num_r=100, num_theta=360)
    x, y, z = create_grid(w, w, res, res, res)
    dose_coords_fixed, doseXYZ_fixed = get_dose_coords(fixed_seeds, dose_LUT, R_LUT, THETA_LUT, x, y, z)
    dose_coords_moving, doseXYZ_moving = get_dose_coords(moving_seeds, dose_LUT, R_LUT, THETA_LUT, x, y, z)
    errorbarXYZ, binaryXYZ = create_errorbar_dose(doseXYZ_fixed, err, res)
    errorbar_coords = grid_to_coords(x, y, z, np.where(errorbarXYZ)).T
    outliers_dict = calc_dose_outliers_convexhull(dose_coords_fixed, moving_seeds)
    outliers_with_errors_dict = calc_dose_outliers_convexhull(errorbar_coords, moving_seeds)
    num_outlier, num_entire_outliers, overlap, max_dist = process_results(dose_coords_fixed,
                                                                          dose_coords_moving,  outliers_dict,)
    plot_results(dose_coords_fixed, fixed_seeds, max_dist, moving_seeds, num_entire_outliers,
                 outliers_dict['outliers'], pair_name, res)

    num_outlier_err, num_entire_outliers_err, overlap_err, max_dist_err = process_results(errorbar_coords, dose_coords_moving,
                                                                            outliers_with_errors_dict,)
    only_error_coords = grid_to_coords(x, y, z, np.where(errorbarXYZ - binaryXYZ)).T

    plot_results(dose_coords_fixed, fixed_seeds, max_dist, moving_seeds, num_entire_outliers,
                 outliers_dict['outliers'], pair_name + "_errorbar", res, other_dose=only_error_coords)
    return (dose_coords_fixed, fixed_seeds, moving_seeds,
    outliers_dict['outliers'], num_outlier, num_entire_outliers,
            num_entire_outliers_err, max_dist, outliers_dict['outliers_dist'], overlap)

def process_results(dose_coords_fixed, dose_coords_moving, outliers_dict):
    outliers, outliers_freq, outlier_dist = (outliers_dict['outliers'],
                                             outliers_dict['outliers_freq'], outliers_dict['outliers_dist'])
    num_outliers = np.sum(outliers_freq > 0)
    num_entire_outliers = np.sum(outliers_freq > 0.95)
    max_dist = np.max(outlier_dist) if len(outlier_dist) else 0
    overlap, max_coords, min_coords = calc_dose_overlap(dose_coords_fixed, dose_coords_moving)
    overlap_ratio = len(overlap.T) / len(dose_coords_moving.T)
    return num_outliers, num_entire_outliers, overlap_ratio, max_dist


def plot_results(dose_coords_fixed, fixed_seeds, max_dist, moving_seeds, num_entire_outliers, outliers, pair_name, res,
                 other_dose=None, other_seed=None):
    if not os.path.exists(f'imgs\{res}'):
        os.makedirs(f'imgs\{res}')
    day1 = pair_name.split('_')[1]
    day2 = pair_name.split('_')[2]
    fig = plot_seeds_and_dose_interactive(dose_coords_fixed, fixed_seeds, color='red', alpha=0.2)
    if other_dose is not None:
        if other_seed is None:
            other_seed = np.zeros((3,3,0))
        fig = plot_seeds_and_dose_interactive(other_dose, other_seed, fig=fig, color='green', alpha=0.05)
    fig = plot_seeds_interactive(moving_seeds, fig=fig, color='blue')
    if len(outliers):
        all_outliers = np.concatenate(outliers, axis=0)
        fig.add_trace(go.Scatter3d(x=all_outliers[:, 0], y=all_outliers[:, 1], z=all_outliers[:, 2], mode='markers',
                                   marker=dict(size=3, color='black')))
    fig.update_layout(title=
                      f' Entire Outliers: {num_entire_outliers},'
                      f' maximum distance: {max_dist:.3f} mm,'
                      )
    name_elements = pair_name.split('_')
    legend_annotations = [
        dict(x=1, y=1, xref='paper', yref='paper',
             text=f'day {day1}', showarrow=False,
             font=dict(color='red')),
        dict(x=1, y=0.95, xref='paper', yref='paper',
             text=f'day {day2}', showarrow=False,
             font=dict(color='blue')),
        dict(x=1, y=0.90, xref='paper', yref='paper',
             text='Outliers', showarrow=False,
             font=dict(color='black')),
    ]
    fig.update_layout(annotations=legend_annotations)
    fig.write_html(f'imgs/{res}/{pair_name}_res_{res}_convex.html')


def combine_results(res1, res2):
    res = []
    for val1, val2 in zip(res1, res2):
        if isinstance(val1, list):
            val1.extend(val2)
        elif isinstance(val1, np.ndarray):
            val1 = np.concatenate((val1, val2), axis=-1)
        else:
            val1 = val1 + val2
        res.append(val1)
    return tuple(res)



def case_experiment(directory, res=0.4, split=True, err_df=None):
    exp_name = directory.split('/')[-1]
    data_dict = read_xlsx(directory)
    res_df = pd.DataFrame({'experiment': [],
                           'num_outliers': [],
                            'num_entire_outliers':[],
                           'num_entire_outliers_err': [],
                           'overlap': [],
                           'outliers_max_dist_mm': [],
                           'outliers_dist_mm':  [],
                           'start_day': [], 'end_day': []})
    # Iterate through each key-value pair in the dictionary
    for key1, val1 in data_dict.items():
        title1, day1, day21 = key1.split('_')
        for key2, val2 in data_dict.items():
            title2, day2, day22 = key2.split('_')
            if key1 != key2 and day1 == day2 and day21 == day22:
                # Found a pair with the same <day> and <day2> but different <title>
                try:
                    if 'fixed' in title1.lower() and 'moving' in title2.lower():
                        print(f'{key1} to {key2}...')
                        pair_name = f'{exp_name}_{day1}_{day21}'
                        if err_df is not None:
                            err = err_df[err_df['case'].apply(lambda x: x == pair_name)]['error_mm'].values[0]
                            # err = min(err, 1)
                        else:
                            err = 1
                        print("error is ", err)
                        if split:
                            val11, val12 = cluster_seeds(val1)
                            val21, val22 = cluster_seeds(val2)
                            res1 = pair_experiment(val11, val21, pair_name + '_1', res=res)
                            res2 = pair_experiment(val12, val22, pair_name + '_2', res=res)
                            (dose_coords_fixed, fixed_seeds, moving_seeds,
                             outliers, num_outliers, num_entire_outliers, num_entire_outliers_err,
                             max_dist, all_dists, overlap) = combine_results(res1, res2)
                        else:
                            (dose_coords_fixed, fixed_seeds, moving_seeds,
                              outliers, num_outliers, num_entire_outliers, num_entire_outliers_err,
                             max_dist,all_dists, overlap) = pair_experiment(val1, val2, pair_name, res=res, err=err)
                        # plot_results(dose_coords_fixed, fixed_seeds, max_dist,
                        #              moving_seeds, num_entire_outliers, outliers, pair_name, res)
                        positive_dists = all_dists[all_dists > 0]
                        res_df = pd.concat([res_df, pd.DataFrame({'experiment': [exp_name],
                                               'num_outliers': [num_outliers],
                                                'num_entire_outliers': [num_entire_outliers],
                                                'num_entire_outliers_err': [num_entire_outliers],
                                                'overlap': [overlap],
                                               'outliers_max_dist_mm': [max_dist],
                                               'outliers_dist_mm':  [positive_dists],
                                               'start_day': [day1], 'end_day': [day21]})], axis=0)
                except (MemoryError, TypeError) as e:
                    print(f'Error processing {key1} and {key2}: {e}')
                    continue
    return res_df

# experiment('data')
if __name__ == '__main__':
    reg_errs = pd.read_csv('regitration_errors.csv')
    for res in [0.3]:
        dirs = os.listdir('data')
        print(dirs)
        tot_df = None
        for d in dirs:
            split=False
            # split = True if d != '13176' else False
            print(f"****Processing {d} with resolution: ", res, "****")
            df = case_experiment(f'data/{d}', res=res, split=split, err_df=reg_errs)
            if tot_df is None:
                tot_df = df
            else:
                tot_df = pd.concat([tot_df, df], axis=0)
            print(tot_df)
        tot_df.to_csv(f'results_res_{res}_convex.csv', index=False)
