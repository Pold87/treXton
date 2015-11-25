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
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

import thread
import time
import threading
from treXton import img_to_texton_histogram
import relocalize

import seaborn as sns

sns.set(style='ticks', palette='Set1')

parser = argparse.ArgumentParser()
parser.add_argument("-tp", "--test_imgs_path", default="/home/pold87/Documents/Internship/datasets/mat/", help="Path to test images")
parser.add_argument("-m", "--mymap", default="../draug/img/bestnewmat.png", help="Path to the mat image")
parser.add_argument("-p", "--predictions", default="predictions.npy", help="Path to the predictions of extract_textons_draug.py")
parser.add_argument("-c", "--camera", default=False, help="Use camera for testing", action="store_true")
parser.add_argument("-mo", "--mode", default=0, help="Use the camera (0), test on train pictures (1), test on test pictures (2)", type=int)
parser.add_argument("-s", "--start_pic", default=950, help="Starting picture (offset)", type=int)
parser.add_argument("-n", "--num_pictures", default=500, help="Amount of pictures for testing", type=int)
parser.add_argument("-ts", "--texton_size", help="Size of the textons", type=int, default=5)
parser.add_argument("-nt", "--num_textons", help="Size of texton dictionary", type=int, default=200)
parser.add_argument("-mt", "--max_textons", help="Maximum amount of textons per image", type=int, default=1000)
parser.add_argument("-tfidf", "--tfidf", default=True, help="Perform tfidf", action="store_false")
parser.add_argument("-std", "--standardize", default=True, help="Perform standarization", action="store_false")
parser.add_argument("-ds", "--do_separate", default=True, help="Use two classifiers (x and y)", action="store_false")
parser.add_argument("-f", "--filter", default=True, help="Use Kalman filter for filtering", action="store_false")
parser.add_argument("-us", "--use_sift", default=True, help="Use SIFT from OpenCV to display its estimation", action="store_false")
parser.add_argument("-ug", "--use_ground_truth", default=False, help="Use SIFT from OpenCV to display its estimation", action="store_true")
args = parser.parse_args()

mymap = args.mymap

def init_tracker():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])

    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])

    tracker.R = np.eye(2) * 1000
    q = Q_discrete_white_noise(dim=2, dt=dt, var=1)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[2000, 0, 700, 0]]).T
    tracker.P = np.eye(4) * 50.
    return tracker


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

def pause_graphs(v, f):
    # as long as the other program did not finish
    while True:
        raw_input("Press Space to pause / continue")
        v.value = (v.value + 1) % 2
        if f == 1:
            break
                
    
    

def show_graphs(v, f):
    

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
    minidrone_f = read_png("img/minidrone_f.png")
    imagebox = OffsetImage(minidrone, zoom=1)
    filter_imagebox = OffsetImage(minidrone_f, zoom=0.7)
    sift_imagebox = OffsetImage(minidrone_f, zoom=0.7)
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
    if args.do_separate:
        clf_x = joblib.load('classifiers/clf_x.pkl')
        clf_y = joblib.load('classifiers/clf_y.pkl') 
    
    else:
        clf0 = joblib.load('classifiers/clf0.pkl')
        clf1 = joblib.load('classifiers/clf1.pkl')
        clfs = [clf0, clf1]

    # Load tfidf
    tfidf = joblib.load('classifiers/tfidf.pkl') 

    if args.mode == 0:
        
        # Initialize camera
        cap = cv2.VideoCapture(0)

        xs = []
        ys = []
        
        i = 0

        mean, stdv = np.load("mean_stdv.npy")

        if args.filter:
            my_filter = init_tracker()

        
        while True:

            # Capture frame-by-frame
            ret, pic = cap.read()
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

            if args.standardize:
                pic = pic - mean
                pic = pic / stdv

            # Get texton histogram of picture
            histogram = img_to_texton_histogram(pic,
                                                kmeans,
                                                args.max_textons,
                                                args.num_textons,
                                                1,
                                                args)

            if args.tfidf:
                histogram = tfidf.transform([histogram]).todense()

            histogram = np.ravel(histogram)


            preds = []
            if args.do_separate:
                pred_x = clf_x.predict([histogram])
                pred_y = clf_y.predict([histogram])
                pred = np.array([[pred_x[0], pred_y[0]]])
                #print("pred x is", pred_x)
                #print("classifier is", clf_x)
            else:
                for clf in clfs:
                    pred = clf.predict([histogram])
                    #print "Pred is",  pred
                    preds.append(pred)

                pred = np.mean(preds, axis=0)
                #print "Averaged pred is", pred
            

            if args.do_separate:
                xy = (pred_x[0], pred_y[0])
            else:
                xy = (pred[0][0], pred[0][1])


            print("xy is", xy)


            if args.filter:
                my_filter.update(pred.T)
                filtered_pred = (my_filter.x[0][0], my_filter.x[2][0])
                my_filter.predict()

                filtered_ab = AnnotationBbox(filter_imagebox, filtered_pred,
                                             xycoords='data',
                                             pad=0.0,
                                             frameon=False)
            

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

        labels = pd.read_csv("handlabeled/playingmat.csv", index_col=0)

        if args.standardize:
            mean, stdv = np.load("mean_stdv.npy")

        if args.filter:
            my_filter = init_tracker()
        
        test_on_the_fly = True
        if test_on_the_fly:
            xs = []
            ys = []

        errors = []
        errors_x = []
        errors_y = []

        # Use SIFT relocalizer from OpenCV/C++
        rel = relocalize.Relocalizer(args.mymap)


        for i in range(args.start_pic, args.start_pic + args.num_pictures, 1):

            while v.value != 0:
                pass

            img_path = path + str(i) + ".png"

            pic = cv2.imread(img_path, 0)

            if args.standardize:
                pic = pic - mean
                pic = pic / stdv


            # Get texton histogram of picture
            histogram = img_to_texton_histogram(pic,
                                                kmeans,
                                                args.max_textons,
                                                args.num_textons,
                                                1,
                                                args)

            if args.tfidf:
                histogram = tfidf.transform([histogram]).todense()

            histogram = np.ravel(histogram)

            # Predict coordinates using supervised learning

            preds = []
            if args.do_separate:
                pred_x = clf_x.predict([histogram])
                pred_y = clf_y.predict([histogram])
                pred = np.array([[pred_x[0], pred_y[0]]])
            else:
                for clf in clfs:
                    pred = clf.predict([histogram])
                    preds.append(pred)

                pred = np.mean(preds, axis=0)

            if args.use_sift:
                sift_loc = rel.calcLocationFromPath(img_path)
                sift_loc[1] = y_width - sift_loc[1]
                #print(sift_loc)
                sift_xy = tuple(sift_loc)

                sift_ab = AnnotationBbox(sift_imagebox, sift_xy,
                                         xycoords='data',
                                         pad=0.0,
                                         frameon=False)

                

            if test_on_the_fly:
                if args.do_separate:
                    xy = (pred_x[0], pred_y[0])
                else:
                    xy = (pred[0][0], pred[0][1])
                
            else:
                xy = (xs[i], ys[i])


            if args.filter:
                my_filter.update(pred.T)
                filtered_pred = (my_filter.x[0][0], my_filter.x[2][0])
                my_filter.predict()

                filtered_ab = AnnotationBbox(filter_imagebox, filtered_pred,
                                             xycoords='data',
                                             pad=0.0,
                                             frameon=False)


                
            
            if args.use_ground_truth:
                ground_truth =  (labels.x[i], labels.y[i])
                diff =  np.subtract(ground_truth, xy)
                abs_diff = np.fabs(diff)
                errors_x.append(abs_diff[0])
                errors_y.append(abs_diff[1])
                error = np.linalg.norm(abs_diff)
                errors.append(error)
                
            if i != args.start_pic:
                
                drone_artist.remove()
                
                if args.filter: filtered_drone_artist.remove()
                if args.use_sift: sift_drone_artist.remove()
                
            ab = AnnotationBbox(imagebox, xy,
            xycoords='data',
            pad=0.0,
            frameon=False)

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
            if args.filter: filtered_drone_artist = ax.add_artist(filtered_ab)
            if args.use_sift: sift_drone_artist = ax.add_artist(sift_ab)

            plt.pause(.8)
                
        if args.use_ground_truth:
            print("errors", np.mean(errors))
            print("errors x", np.mean(errors_x))
            print("errors y", np.mean(errors_y))
    else:
        print("Unknown mode; Please specify a mode (0, 1, 2)")



if __name__ == "__main__":
    v = Value('i', 0)
    global f
    f = Value('f', 0)

    global user_paused
    user_paused = False

    p1 = Process(target=show_graphs, args=(v, f))
    p1.start()
    pause_graphs(v, f)
    p1.join()
    p2.join()

