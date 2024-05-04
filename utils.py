import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import os, glob
import pydicom
# from dicom_contour.contour import *
import cv2
import pandas as pd
from scipy import interpolate
from skimage import draw
import matplotlib.cm as cm
from glob import glob
from numpy.linalg import norm
import zipfile
import pathlib
import plotly.graph_objs as go
from plotly.subplots import make_subplots


FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2


SEED_DIAMETER = 0.35
SEED_LENGTH_MM = 10
CONTOURS_TO_DISPLAY = {'GTV', 'CTV', 'PTV'}






def display_axial(arr, contours=None, seeds=None, aspect=1, ax=None):
    """
    display axial plane
    :param arr: array of data
    :param contours: array of contours
    :param seeds: array of seeds
    :param aspect: aspect ratio
    :param ax: matplotlib ax object (if None, a new one wil be created)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    masked_contour_arr = np.ma.masked_where(contours == 0, contours) if contours is not None else None
    # masked_seed_arr = np.ma.masked_where(seeds == 255, seeds) if seeds is not None else None
    tracker = IndexTracker(ax, arr, masked_contour_arr, seeds, aspect)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key)

    plt.show()


def display_saggital(arr, contours, seeds, aspect):
    """
    display saggital plane
    :param arr: array of data
    :param contours: array of contours
    :param seeds: array of seeds
    :param aspect: aspect ratio
    :param ax: matplotlib ax object (if None, a new one wil be created)
    """
    arr = np.swapaxes(arr, 1, 2)
    contours = np.swapaxes(contours, 1, 2) if contours is not None else None
    display_axial(arr, contours, seeds, aspect)


def display_coronal(arr, contours, seeds, aspect):
    """
    display coronal plane
    :param arr: array of data
    :param contours: array of contours
    :param seeds: array of seeds
    :param aspect: aspect ratio
    :param ax: matplotlib ax object (if None, a new one wil be created)
    """
    arr = np.swapaxes(arr, 0, 2)
    contours = np.swapaxes(contours, 0, 2) if contours is not None else None
    display_axial(arr, contours, seeds, aspect)

def poly_area2D(poly):
    """
    calculate 2d polygon area
    :param poly: points cloud of the perimeter of the polygon
    :return: area of polygon
    """
    total = 0.0
    N = len(poly)
    for i in range(N):
        v1 = poly[i]
        v2 = poly[(i+1) % N]
        total += v1[0]*v2[1] - v1[1]*v2[0]
    return abs(total/2)


def calc_poly_vol(arr, slice_thickness):
    """
    calculate 3d polygon volume
    :param arr: points cloud of the perimeter of the polygon
    :return:
    """
    vol = 0
    z_vals = np.unique(arr[:, 2])
    for z in z_vals:
        idx = np.where(arr[:,2] == z)[0]
        vol += poly_area2D(arr[idx,:2])*abs(slice_thickness)
    return vol


def normalize(vec):
    n = norm((vec))
    if n == 0:
        return vec
    return vec / n


def draw_circle_3d(start_point, end_point, radius):
    direction = normalize(end_point - start_point)
    zeros_in_direction = np.where(direction == 0)[0]
    vec1, vec2 = np.zeros_like(direction), np.zeros_like(direction)
    if len(zeros_in_direction) == 2:
        vec1[zeros_in_direction[0]] = 1
        vec2[zeros_in_direction[1]] = 1
    else:
        sorted_direction = np.argsort(direction)
        vec1 = np.zeros_like(direction)
        vec1[sorted_direction[1]] = -direction[sorted_direction[2]]
        vec1[sorted_direction[2]] = direction[sorted_direction[1]]
        vec1 = normalize(vec1)
        # vec1 = normalize(np.array([0,0,-direction[2]])) if 2 not in zeros_in_direction else normalize(np.array([-direction[0],0,0]))
        vec2 = normalize(np.cross(direction, vec1))
    try:
        assert np.dot(direction, vec1) == np.dot(vec1, vec2) == np.dot(direction, vec2) == 0
    except AssertionError:
        pass

    # print("spanning vector are not perpendicular. dot is ", np.dot(direction, vec1), np.dot(vec1, vec2), np.dot(direction, vec2))
    x_ = np.linspace(start_point[0], end_point[0], 30)
    y_ = np.linspace(start_point[1], end_point[1], 30)
    z_ = np.linspace(start_point[2], end_point[2], 30)
    x, y, z = np.meshgrid(x_, y_, z_)
    mesh = np.array((x,y,z))
    theta = np.linspace(0, 2*np.pi, 30)[...,None]
    circle = start_point + radius*np.cos(theta)*(vec1[None,...]) + radius*np.sin(theta)*(vec2[None, ...])
    return vec1, vec2, circle

def get_seed_coords(seed, max_r, dr):
    direction = normalize(seed[2, :] - seed[0, :])
    start = seed[0, :].astype(np.float64)
    end = seed[2, :]
    seed_cords = np.zeros((0, 3))
    while np.sqrt(np.sum(start - end) ** 2) > 1:
        radiuses = np.arange(0, int(np.round(max_r)), dr)
        disk = np.zeros((0, 3))
        for r in radiuses:
            vec1, vec2, circle = draw_circle_3d(start, end, r)
            disk = np.vstack([disk, circle])
        seed_cords = np.vstack([seed_cords, disk])
        start += direction * 0.5
    return seed_cords

def get_spacing(meta):
    """
    get spacing array - [x pixel spacing, y pixel spacing, z pixel spacing]
    :param meta: meta data dict
    :return: spacing array
    """
    try:
        spacing = np.array([meta['pixelSpacing'][0],
                            meta['pixelSpacing'][1], abs(meta['sliceSpacing'])])
    except Exception as e:
        # print("spacing array ", e)
        spacing = np.array([meta['pixelSpacing'][0],
                            meta['pixelSpacing'][1], abs(meta['sliceThickness'])])
    return spacing


def read_structure(dir, slice_spacing):
    """
    read contours
    :param dir: directory to folder with dcm file of contours
    :return: list of tuples. each tuple contains (name, arr). name is contour name (string), arr is contour data
    ((3,N) nd-array) of the perimeter voxels of the contour
    """
    for f in glob("%s/*.dcm" % dir):

        # filename = f.split("/")[-1]
        ds = pydicom.dcmread(f)
        # print(ds)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ctrs = ds.ROIContourSequence
        meta = ds.StructureSetROISequence
        # print(ds)
        list_ctrs = []
        for i in range(len(ctrs)):
            data = ctrs[i].ContourSequence
            name = meta[i].ROIName
            vol = 0
            arr = np.zeros((3,0))
            for j in range(len(data)):
                contour = data[j].ContourData
                np_contour = np.zeros((3, len(contour) // 3))
                for k in range(0, len(contour), 3):
                    np_contour[:, k // 3] = contour[k], contour[k + 1], contour[k + 2]
                # if data[j].ContourGeometricType == "CLOSED_PLANAR":
                    # vol += calc_poly_area(np_contour)
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.plot(np_contour[0], np_contour[1], np_contour[2])
                # plt.show()
                vol += poly_area2D(np_contour.T) * abs(slice_spacing)
                arr = np.hstack((arr, np_contour))
            # vol /= 2 # compatible with MIM
            vol /= 1000 # convert mm**3 to ml (cm**3)
            # print(name, " volume ", vol)
            list_ctrs.append((name, arr, vol))
        return list_ctrs


def read_structure_from_csv(path):
    """
    read structure from csv. csv is created by MIM workflow
    :param path: path to csv file
    :return: (3,N) nd array - each row in array is a voxel contained inside the contour
    """
    df = pd.read_csv(path)
    arr = df.to_numpy()
    return arr[:, :3].T



def plot_seeds(seeds_tips, ax: object = None, title: object = None, color: object = None) -> object:
    """
    plot seeds
    :param seeds_tips: (3,3,N) array of seeds tips
    :param ax: matplotlib axes object (if None, a new one will be created)
    :param title: title of the plit
    :param color: color of seeds (if not specified, each seeds will be in different color)
    :return: ax object contains seeds plot. this can be showed using plt.show() command
    """
    if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
    for i in range(seeds_tips.shape[-1]):
        x = seeds_tips[:, 0, i]
        y = seeds_tips[:, 1, i]
        z = seeds_tips[:, 2, i]
        if color is None:
            ax.plot(x, y, z, label="seed number %d" % (i + 1))
        else:
            ax.plot(x, y, z, color=color)

        if title is not None:
            plt.title(title)
        # plt.legend()
    return ax

def plot_seeds_and_dose(dose_coords, seeds, ax=None, color=None, alpha=0.1):
    """

    :param dose_coords: 3XN array of dose coordinates
    :param seeds: 3X3XN array of seeds
    :param ax:
    :return:
    """
    if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
    ax = plot_seeds(seeds, ax, color=color)
    if color is None:
        ax.scatter(dose_coords[0], dose_coords[1], dose_coords[2], marker='o', alpha=alpha)
    else:
        ax.scatter(dose_coords[0], dose_coords[1], dose_coords[2],color=color, marker='o', alpha=alpha)
    return ax


def plot_seeds_interactive(seeds_tips, fig=None, title=None, color=None):
    if fig is None:
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    for i in range(seeds_tips.shape[-1]):
        x = seeds_tips[:, 0, i]
        y = seeds_tips[:, 1, i]
        z = seeds_tips[:, 2, i]
        if color is None:
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name="seed number %d" % (i + 1)))
        else:
            fig.add_trace(
                go.Scatter3d(x=x, y=y, z=z, mode='lines', name="seed number %d" % (i + 1), line=dict(color=color)))

    fig.update_layout(title=title)
    return fig


def plot_seeds_and_dose_interactive(dose_coords, seeds, fig=None, color=None, alpha=0.1):
    if fig is None:
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    fig = plot_seeds_interactive(seeds, fig, color=color)

    if color is None:
        fig.add_trace(go.Scatter3d(x=dose_coords[0], y=dose_coords[1], z=dose_coords[2], mode='markers',
                                   marker=dict(size=3, opacity=alpha)))
    else:
        fig.add_trace(go.Scatter3d(x=dose_coords[0], y=dose_coords[1], z=dose_coords[2], mode='markers',
                                   marker=dict(size=3, color=color, opacity=alpha)))

    return fig
def plot_seeds_together(seeds1, seeds2, indexes, excludes, ax=None, title=None):
    """
    plot seeds from different plans together
    :param seeds1: (3,3,N) array of seeds from first plan
    :param seeds2: (3,3,N) array of seeds from second plan
    :param indexes: indexes of seeds in the first plan
    :param excludes: indexes to exclude (they will not be plotted)
    :param ax: matplotlib axes object (if None, a new one will be created)
    :param title: title for the plot
    :return: ax object contains seeds plot. this can be showed using plt.show() command
    """
    start_on_legend = False
    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    colors = plt.cm.get_cmap("gist_rainbow", max(seeds1.shape[-1], seeds2.shape[-1]))(range(max(seeds1.shape[-1], seeds2.shape[-1])))
    # colors = cm.rainbow(np.linspace(0, 1, max(seeds1.shape[-1], seeds2.shape[-1])))
    for i in range(max(seeds1.shape[-1], seeds2.shape[-1])):
        x1, y1, z1 = seeds1[:, 0, i], seeds1[:, 1, i], seeds1[:, 2, i]
        x2, y2, z2 = seeds2[:, 0, i], seeds2[:, 1, i], seeds2[:, 2, i]
        start1 = seeds1[0,:,i]
        start2 = seeds2[0,:,i]

        # if color is None:
        color = colors[i]
        if indexes[i] not in excludes:
            ax.plot(x1, y1, z1, label="seed number %d" % (indexes[i] + 1), color="r", linewidth=1)
            ax.plot(x2, y2, z2, color="b", linewidth=1, linestyle='dashed')
            # if not start_on_legend:
            #     ax = add_point_on_plot(ax, start1, "black", "start of seed")
            #     start_on_legend = True
            # else:
            #     ax = add_point_on_plot(ax, start1, "black")
            # ax = add_point_on_plot(ax, start2, "black")

        # else:
        #     ax.plot(x1, y1, z1, color=color[0])
        #     ax.plot(x2, y2, z2, color=color[1])
    if title is not None:
        plt.title(title)
    return ax

def add_point_on_plot(ax, point, color='black', label=''):
    ax.scatter(point[0], point[1], point[2], color=color, label=label)
    return ax


def plot_seeds_from_mask(seeds_mask, num_labels, ax=None):
    """
    plot seeds from mask array. the array is labeled according to seed's number
    :param seeds_mask: mask array
    :param num_labels: number of labels
    :param ax: matplotlib axes object
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    for i in range(num_labels):
        x = np.where(seeds_mask == i + 1)[0]
        y = np.where(seeds_mask == i + 1)[1]
        z = np.where(seeds_mask == i + 1)[2]
        ax.plot(x, y, z, label="seed number %d" % (i + 1))
    plt.show()


def plot_contour(contours, name, ax=None):
    """
    plot contours
    :param contours: (3,N) array represent x,y,z coordinated of N points
    :param name: name of contour (for legend)
    :param ax: matplotlib axes object (if None, a new one will be created)
    :return: ax object contains seeds plot. this can be showed using plt.show() command
    """
    # ax = None
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    x,y,z = contours[0], contours[1], contours[2]
    ax.scatter3D(x,y,z, label=name, alpha=0.01)
    # ax.plot_trisurf(x, y, z, alpha=0.1, label=name)
    return ax


def draw_seeds_2d(img, seed_tips, color):
    for s in range(seed_tips.shape[-1]):
        for j in [0,2]:
            x,y,z = seed_tips[j,1,s], seed_tips[j,0,s], seed_tips[j,2,s]
            rr,cc = draw.circle(x, y, 5)
            in_bound_idx = np.bitwise_and(rr < img.shape[0], cc < img.shape[1])
            img[rr[in_bound_idx],cc[in_bound_idx],int(round(z))] = color
    return img

def create_seeds_mask(images, seeds_tips, seeds_orientation):
    mask = np.zeros_like(images)
    for i in range(seeds_tips.shape[-1]):
        s = seeds_tips[0, :, i].astype(np.int32)
        e = seeds_tips[2, :, i].astype(np.int32)
        while sum(abs(s - e)) > 1:
            r, c, z = s[1], s[0], s[2]
            mask[r, c, z] = i + 1
            s = np.round(s + seeds_orientation[:, i]).astype(np.int32)
    return mask

def draw_seeds_mask(img, seeds_position, seed_orientation):
    # mask = np.zeros(img.shape)
    # img = convert_to_rgb(img)
    # fig, ax = plt.subplots(1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(seeds_position.shape[1]):
        x,y,z = seeds_position[:3, i]
        u,v,w = seed_orientation[:3, i]
        ax.quiver(x,y,z,u,v,w)
        # col, row = seeds_position[:2, i]
        # slice = seeds_position[2, i]
        # circ = mpatches.Circle((col, row), 4)
        # ax.add_patch(circ)
    #     mask[..., round(slice)] = cv2.circle(mask[..., round(slice)].astype(np.int32), (round(row), round(col)), 10, [255, 0,0], thickness=2)
    #     # ax.imshow(mask[..., round(slice)], cmap='gray')
    #     # plt.show()
    plt.show()
    # return mask

def display_seeds_and_contours(struct, seed_tips):
    fig = None
    for name, coords in struct:
        fig = plot_contour(coords, name, fig)
    fig = plot_seeds(seed_tips, fig)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.zlabel('slice')
    # fig.legend()
    plt.show()


class IndexTracker(object):
    def __init__(self, ax, X, contours=None, seeds=None, aspect=1):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')

        self.X = X
        self.contours = contours
        self.seeds = seeds
        if len(X.shape) == 3:
            rows, cols, self.slices = X.shape
            self.channels = 0
        else:
            rows, cols, self.channels, self.slices = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[..., self.ind], cmap='gray')
        self.struct = ax.imshow(self.contours[..., self.ind], cmap='cool', interpolation='none',
                                alpha=0.7) if self.contours is not None else None
        # draw_seeds_mask(ax, X, seeds_position)
        self.plan = ax.imshow(self.seeds[..., self.ind]) if self.seeds is not None else None
        ax.set_aspect(aspect)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def on_key(self, event):
        if event.key == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[..., self.ind])
        if self.contours is not None:
            self.struct.set_data(self.contours[:,:, self.ind])
        if self.seeds is not None:
            self.plan.set_data(self.seeds[:,:, self.ind])
        self.ax.set_ylabel('Slice Number: %s' % self.ind)
        # cid = self.ax.canvas.mpl_connect('key_press_event', self.on_key)
        self.im.axes.figure.canvas.draw()

    def zoom_in(self, x,y):
        print("now i should zoom in around ", x , y)

    def zoom_out(self, x, y):
        print("now i should zoom out around ", x, y)



os.system("tree C:/Users/Dicom_ROI")



