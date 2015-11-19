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
import argparse
from multiprocessing import Process, Value

import thread
import time
import threading

import seaborn as sns

sns.set(style='ticks', palette='Set1')

parser = argparse.ArgumentParser()
parser.add_argument("-tp", "--test_imgs_path", default="/home/pold/Documents/datasets/mat/", help="Path to test images")
parser.add_argument("-m", "--mymap", default="../draug/img/bestnewmat.png", help="Path to the mat image")
parser.add_argument("-p", "--predictions", default="predictions.npy", help="Path to the predictions of extract_textons_draug.py")

args = parser.parse_args()

mymap = args.mymap


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

def pause_graphs(v):
    while True:
        raw_input("Press Space to pause / continue")
        v.value = (v.value + 1) % 2
    

def show_graphs(v):
    

    plt.ion()
    predictions = np.load(args.predictions)
    path = args.test_imgs_path
    background_map = plt.imread(mymap)
    y_width, x_width, _ = background_map.shape
    
    # First set up the figure, the axis, and the plot element we want to animate
    ax = plt.subplot2grid((2,2), (0, 0))
    ax.set_xlim([0, x_width])
    ax.set_ylim([-y_width / 2, y_width / 2])

    line, = ax.plot([], [], lw=2)

    ax.set_title('Position prediction based on textons')

    xs = predictions[:, 0]
    ys = predictions[:, 1]

    start_pic = 30

    minidrone = read_png("img/minidrone.png")
    imagebox = OffsetImage(minidrone, zoom=1)
    ax.imshow(background_map, zorder=0, extent=[0, x_width, -y_width / 2, y_width / 2])

    optitrack = np.load("optitrack_coords.npy")
    ax_opti = plt.subplot2grid((2,2), (1, 0), colspan=2)
    ax_opti.set_title('OptiTrack ground truth')
    line_opti, = ax_opti.plot([], [], lw=2)
    ax_opti.set_xlim([-10, 10])
    ax_opti.set_ylim([-10, 10])


    xs_opti = optitrack[:, 0]
    ys_opti = optitrack[:, 1]
    ys_opti, xs_opti = rotate_coordinates(xs_opti, ys_opti, np.radians(37))

    ax_inflight = plt.subplot2grid((2,2), (0, 1))
    ax_inflight.set_title('Pictures taken during flight')

    for i in range(start_pic, len(xs)):

        while v.value != 0:
            pass
        
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
    v = Value('i', 0)

    global user_paused
    user_paused = False

    p1 = Process(target=show_graphs, args=(v, ))
    p1.start()
    pause_graphs(v)
    p1.join()

