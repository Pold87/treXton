import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import scipy
import cv2
from matplotlib.cbook import get_sample_data
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
    AnnotationBbox
from matplotlib._png import read_png

import seaborn as sns
sns.set(style='ticks', palette='Set1')

def rotate_coordinates(xs, ys, theta):

    """
    This function rotates 2D coordinates by a specified angle.
    """
    
    xs_new = np.zeros(len(xs))
    ys_new = np.zeros(len(ys))
    
    for i, (x, y) in enumerate(zip(xs, ys)):

        new_x = np.cos(theta) * x - np.sin(theta) * y
        new_y = np.sin(theta) * x + np.cos(theta) * y

        xs_new[i] = new_x
        ys_new[i] = new_y

    return xs_new, ys_new



def show_graphs():
    

    plt.ion()
    predictions = np.load("filtered.npy")
    path = "/home/pold87/Downloads/imgs_first_flight/"

    # First set up the figure, the axis, and the plot element we want to animate
    ax = plt.subplot2grid((2,2), (0, 0))
    ax.set_xlim([0, 4300])
    ax.set_ylim([-2343, 2343])

    line, = ax.plot([], [], lw=2)

    ax.set_title('Predictions')

    xs = predictions[:, 0]
    ys = predictions[:, 1]

    start_pic = 30

    minidrone = read_png("img/minidrone.png")
    imagebox = OffsetImage(minidrone, zoom=1)
    background_map = plt.imread("../draug/img/bestmap.png")
    ax.imshow(background_map, zorder=0, extent=[0, 4300, -2343, 2343])

    optitrack = np.load("optitrack_coords.npy")
    ax_opti = plt.subplot2grid((2,2), (1, 0), colspan=2)
    line_opti, = ax_opti.plot([], [], lw=2)
    ax_opti.set_xlim([-10, 10])
    ax_opti.set_ylim([-10, 10])


    xs_opti = optitrack[:, 0]
    ys_opti = optitrack[:, 1]
    ys_opti, xs_opti = rotate_coordinates(xs_opti, ys_opti, np.radians(37))

    ax_inflight = plt.subplot2grid((2,2), (0, 1))

    for i in range(start_pic, len(xs)):
        img_path = path + str(i) + ".jpg"

        xy = (xs[i], ys[i])

        if i != start_pic:
            drone_artist.remove()
        ab = AnnotationBbox(imagebox, xy,
        xycoords='data',
        pad=0.0,
        frameon=False)

        line.set_xdata(xs[max(start_pic, i - 13):i])  # update the data
        line.set_ydata(ys[max(start_pic, i - 13):i])

        line_opti.set_xdata(xs_opti[max(start_pic, i - 25):i])  # update the data
        line_opti.set_ydata(ys_opti[max(start_pic, i - 25):i])

        pic = mpimg.imread(img_path)
        if i == start_pic:
            img_artist = ax_inflight.imshow(pic)
        else:
            img_artist.set_data(pic)

        drone_artist = ax.add_artist(ab)

        plt.pause(.001)


if __name__ == "__main__":
    show_graphs()            

