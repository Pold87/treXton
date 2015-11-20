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
from sklearn.externals import joblib

import thread
import time
import threading
from treXton import img_to_texton_histogram

import seaborn as sns

sns.set(style='ticks', palette='Set1')

parser = argparse.ArgumentParser()
parser.add_argument("-tp", "--test_imgs_path", default="/home/pold87/Documents/Internship/draug/genimgs/", help="Path to test images")
parser.add_argument("-m", "--mymap", default="../draug/img/bestnewmat.png", help="Path to the mat image")
parser.add_argument("-p", "--predictions", default="predictions.npy", help="Path to the predictions of extract_textons_draug.py")
parser.add_argument("-c", "--camera", default=False, help="Use camera for testing", action="store_true")
parser.add_argument("-mo", "--mode", default=0, help="Use the camera (0), test on train pictures (1), test on test pictures (2)", type=int)
parser.add_argument("-s", "--start_pic", default=950, help="Starting picture (offset)", type=int)
parser.add_argument("-n", "--num_pictures", default=500, help="Amount of pictures for testing", type=int)
parser.add_argument("-ts", "--texton_size", help="Size of the textons", type=int, default=5)
parser.add_argument("-nt", "--num_textons", help="Size of texton dictionary", type=int, default=100)
parser.add_argument("-mt", "--max_textons", help="Maximum amount of textons per image", type=int, default=500)


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
    ax.set_ylim([0, y_width])

    line, = ax.plot([], [], lw=2)

    ax.set_title('Position prediction based on textons')

    xs = predictions[:, 0]
    ys = predictions[:, 1]

    minidrone = read_png("img/minidrone.png")
    imagebox = OffsetImage(minidrone, zoom=1)
    ax.imshow(background_map, zorder=0, extent=[0, x_width, 0, y_width])


    if args.mode == 0:
        ax_opti = plt.subplot2grid((2,2), (1, 0), colspan=2)
        ax_opti.set_title('Texton histogram')
        line_opti, = ax_opti.plot([], [], lw=2)

    elif args.mode == 1:
        ax_opti = plt.subplot2grid((2,2), (1, 0), colspan=2)
        ax_opti.set_title('Texton histogram')
        line_opti, = ax_opti.plot([], [], lw=2)

        optitrack = pd.read_csv("../draug/targets.csv")
        xs_opti = optitrack.x
        ys_opti = optitrack.y
        
        
    elif args.mode == 2:
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

    # Load k-means
    kmeans = joblib.load('classifiers/kmeans.pkl') 
        
    # Load random forest
    clf = joblib.load('classifiers/randomforest.pkl') 

    # Load tfidf
    tfidf = joblib.load('classifiers/tfidf.pkl') 


    if args.mode == 0:
        
        # Initialize camera
        cap = cv2.VideoCapture(0)

        xs = []
        ys = []
        
        i = 0
        while True:

            # Capture frame-by-frame
            ret, pic = cap.read()
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

            # Get texton histogram of picture
            histogram = img_to_texton_histogram(pic,
                                                kmeans,
                                                args.max_textons,
                                                args.num_textons,
                                                1,
                                                args)

            histogram = tfidf.transform([histogram]).todense()
            histogram = histogram[0]

            # Predict coordinates using supervised learning
            pred = clf.predict([histogram])
            x = pred[0][0]
            y = pred[0][1]
            xs.append(x)
            ys.append(y)
            
            xy = (x, y)

            # Update predictions graph
            line.set_xdata(xs[max(0, i - 13):i]) 
            line.set_ydata(ys[max(0, i - 13):i])
            
            ab = AnnotationBbox(imagebox, xy,
                                xycoords='data',
                                pad=0.0,
                                frameon=False)

            if i == 0:
                histo_bar = ax_opti.bar(np.arange(len(histogram)), histogram)
                img_artist = ax_inflight.imshow(pic)
            else:
                img_artist.set_data(pic)
                drone_artist.remove()
                
                for rect, h in zip(histo_bar, histogram):
                    rect.set_height(h)
    
            drone_artist = ax.add_artist(ab)

            plt.pause(.01)
            
            i += 1
        
    elif args.mode == 1 or args.mode == 2:

        test_on_the_fly = True
        if test_on_the_fly:
            xs = []
            ys = []

        for i in range(args.start_pic, args.start_pic + args.num_pictures, 2):

            while v.value != 0:
                pass

#            if args.mode == 1:
#            img_path = path + str(i) + ".png"
 #           else:
                

            img_path = path + str(i) + ".png"

            pic = cv2.imread(img_path, 0)

            # Get texton histogram of picture
            histogram = img_to_texton_histogram(pic,
                                                kmeans,
                                                args.max_textons,
                                                args.num_textons,
                                                1,
                                                args)

            histogram = tfidf.transform([histogram]).todense()
            print(histogram.shape)

            histogram = np.ravel(histogram)

            # Predict coordinates using supervised learning
            print(histogram)
            pred = clf.predict([histogram])

            print("Ground truth (x, y)", xs_opti[i], ys_opti[i])
            print("Prediction (x, y)", pred[0][0], pred[0][1])

            if test_on_the_fly:
                xy = (pred[0][0], pred[0][1])
            else:
                xy = (xs[i], ys[i])

            if i != args.start_pic:
                drone_artist.remove()
            ab = AnnotationBbox(imagebox, xy,
            xycoords='data',
            pad=0.0,
            frameon=False)


            print("Image path", img_path)
 
            if test_on_the_fly:
                pass
            else:
                # Update predictions graph
                line.set_xdata(xs[max(args.start_pic, i - 13):i + 1]) 
                line.set_ydata(ys[max(args.start_pic, i - 13):i + 1])

                # Update optitrack graph
                print(xs[max(args.start_pic, i - 25):i])

#                line_opti.set_xdata(xs_opti[i])  # update the data
#                line_opti.set_ydata(ys_opti[i])
                #line_opti.set_array([xs_opti[i], ys_opti[i]])  # update the data

            if args.mode == 2:
                line_opti.set_xdata(xs_opti[i])  # update the data
                line_opti.set_ydata(ys_opti[i])

            if i == args.start_pic:
                img_artist = ax_inflight.imshow(pic)
            else:
                img_artist.set_data(pic)

            if args.mode == 1 or args.mode == 2:
                if i == args.start_pic:
                    histo_bar = ax_opti.bar(np.arange(len(histogram)), histogram)
                else:
                    for rect, h in zip(histo_bar, histogram):
                        rect.set_height(h)


            drone_artist = ax.add_artist(ab)

            plt.pause(.8)

    else:
        print("Unknown mode; Please specify a mode (0, 1, 2)")


if __name__ == "__main__":
    v = Value('i', 0)

    global user_paused
    user_paused = False

    p1 = Process(target=show_graphs, args=(v, ))
    p1.start()
    pause_graphs(v)
    p1.join()

