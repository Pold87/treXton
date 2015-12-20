from __future__ import division

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
from treXton import img_to_texton_histogram, RGB2Opponent, imread_opponent
import relocalize
import configargparse
import treXtonConfig
import seaborn as sns
from treXtonConfig import parser
import time
import particle_filter as pf
import xgboost as xgb

sns.set(style='ticks', palette='Set1')

args = parser.parse_args()
mymap = args.mymap

def mydist(x, y):

    x_norm = x / np.sum(x)
    y_norm = y / np.sum(y)

    return np.sum((x_norm - y_norm) ** 2 / (x_norm + y_norm)) / 2


def pred_ints(model, X, percentile=60):
    err_down = []
    err_up = []
    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(X)[0])
    err_down = np.percentile(preds, (100 - percentile) / 2. )
    err_up = np.percentile(preds, 100 - (100 - percentile) / 2.)
    return err_down, err_up


def init_tracker():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])

    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])

    tracker.R = np.eye(2) * 5
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[-3, 0, 229, 0]]).T
    tracker.P = np.eye(4) * 5.
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
    minidrone_s = read_png("img/minisift.png")
    imagebox = OffsetImage(minidrone, zoom=1)
    filter_imagebox = OffsetImage(minidrone_f, zoom=0.6)
    sift_imagebox = OffsetImage(minidrone_s, zoom=0.7)
    ax.imshow(background_map, zorder=0, extent=[0, x_width, 0, y_width])


    if args.mode == 0:
        ax_opti = plt.subplot2grid((2,2), (1, 0), colspan=2)
        ax_opti.set_title('Texton histogram')
        line_opti, = ax_opti.plot([], [], lw=2)

    elif args.mode == 1:
        ax_opti = plt.subplot2grid((2,2), (1, 0), colspan=2)
        ax_opti.set_title('Texton histogram')
        line_opti, = ax_opti.plot([], [], lw=2)

        #optitrack = pd.read_csv("../draug/targets.csv")
        #xs_opti = optitrack.x
        #ys_opti = optitrack.y
        
        
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

    kmeans = []
    for channel in range(3):
    
        kmean = joblib.load('classifiers/kmeans' + str(channel) + '.pkl')
        kmeans.append(kmean)

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
        cap = cv2.VideoCapture(args.dev)

    labels = pd.read_csv("handlabeled/playingmat.csv", index_col=0)

    xs = []
    ys = []
        
    i = 0

    if args.filter:
        my_filter = init_tracker()
            
        
    xs = []
    ys = []

    errors = []
    errors_x = []
    errors_y = []

    # Use SIFT relocalizer from OpenCV/C++
    if args.use_sift:
        rel = relocalize.Relocalizer(args.mymap)


    labels = pd.read_csv("handlabeled/playingmat.csv", index_col=0)

    if args.use_ground_truth:
        truth = pd.read_csv("../datasets/imgs/sift_targets.csv")
        truth.set_index(['id'], inplace=True)

    if args.use_particle_filter:
        mydrone = pf.robot()
        N = 80  # Number of particles
        p = pf.init_particles(N)
        dt = 1

    while True:

        start = time.time()

        while v.value != 0:
            pass
        
        if args.mode == 0:
            # Capture frame-by-frame
            ret, pic = cap.read()
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
            pic = RGB2Opponent(pic)

        else:
            img_path = path + str(i) + ".png"
            pic_c = imread_opponent(img_path)
            pic = imread_opponent(img_path)


        if args.standardize:
            for channel in range(args.channels):
                mean, stdv = np.load("mean_stdv_" + str(channel) + ".npy")
                pic[:, :, channel] = pic[:, :, channel] - mean
                pic[:, :, channel] = pic[:, :, channel] / stdv

        if args.local_standardize:

            mymean, mystdv = cv2.meanStdDev(pic)
            mymean = mymean.reshape(-1)
            mystdv = mystdv.reshape(-1)

            pic = (pic - mymean) / mystdv         
            

        # Get texton histogram of picture
        query_histograms = np.zeros((args.channels, args.num_textons))

        if args.color_standardize:

            mymean = np.mean(np.ravel(pic[:, :, 0]))
            mystdv = np.std(np.ravel(pic[:, :, 0]))

            pic[:, :, 0] = pic[:, :, 0] - mymean
            pic[:, :, 0] = pic[:, :, 0] / mystdv
            pic[:, :, 1] = pic[:, :, 1] / mystdv
            pic[:, :, 2] = pic[:, :, 2] / mystdv

        if args.use_dipoles:
            histogram = img_to_texton_histogram(pic,
                                                kmeans[0],
                                                args.max_textons,
                                                args.num_textons,
                                                1,
                                                args,
                                                0)
            query_histograms = histogram.reshape(1, args.num_textons * 4)
                
        else:
            for channel in range(args.channels):
                histogram = img_to_texton_histogram(pic[:, :, channel],
                                                        kmeans[channel],
                                                        args.max_textons,
                                                        args.num_textons,
                                                        1,
                                                        args,
                                                        channel)
                query_histograms[channel] = histogram

            query_histograms = query_histograms.reshape(1, args.num_textons * args.channels)            
                             

        if args.tfidf:
            query_histograms = tfidf.transform(query_histograms).todense()
            histogram = np.ravel(histogram)


        preds = []
        if args.do_separate:

            if args.use_xgboost:
                dtest = xgb.DMatrix(query_histograms)
                pred_x = clf_x.predict(dtest)
                pred_y = clf_y.predict(dtest)
            else:                
                pred_x = clf_x.predict(query_histograms)
                pred_y = clf_y.predict(query_histograms)


            #err_down_x, err_up_x = pred_ints(clf_x, [histogram])
            #err_down_y, err_up_y = pred_ints(clf_y, [histogram])

            #err_x = pred_x - err_down_x
            #err_y = pred_y - err_down_y

            pred = np.array([[pred_x[0], pred_y[0]]])
            #print("pred x is", pred_x)
            #print("classifier is", clf_x)
            xy = (pred_x[0], pred_y[0])
        else:
            for clf in clfs:
                pred = clf.predict(query_histograms)
                #print "Pred is",  pred
                preds.append(pred)

                pred = np.mean(preds, axis=0)
                #print "Averaged pred is", pred
            xy = (pred[0][0], pred[0][1])

        # Pritn prediction that is used for plotting
        print(xy)


        # Get particle positions
        if args.use_particle_filter:
            p = pf.move_all(p, xy, dt)
            plt_xs, plt_ys = pf.get_x_y(p)        

        if args.use_sift:
            #sift_loc = rel.calcLocationFromPath(img_path)
            #sift_loc[1] = y_width - sift_loc[1]
            #print(sift_loc)
            #sift_xy = tuple(sift_loc)
            #sift_x = truth.ix[i, "x"]
            #sift_y = truth.ix[i, "y"]
            sift_xy = (sift_x, sift_y)

            sift_ab = AnnotationBbox(sift_imagebox, sift_xy,
                                     xycoords='data',
                                     pad=0.0,
                                     frameon=False)


        if args.use_normal:
            ab = AnnotationBbox(imagebox, xy,
                                xycoords='data',
                                pad=0.0,
                                frameon=False)


        if args.filter:
            my_filter.update(pred.T)
            filtered_pred = (my_filter.x[0][0], my_filter.x[2][0])
            my_filter.predict()
            
            filtered_ab = AnnotationBbox(filter_imagebox, filtered_pred,
                                         xycoords='data',
                                         pad=0.0,
                                         frameon=False)
            

        if args.use_ground_truth:
            ground_truth =  (truth.ix[i, "x"], truth.ix[i, "y"])
            diff =  np.subtract(ground_truth, xy)
            abs_diff = np.fabs(diff)
            errors_x.append(abs_diff[0])
            errors_y.append(abs_diff[1])
            error = np.linalg.norm(abs_diff)
            errors.append(error)
            
            
            # Update predictions graph
            line.set_xdata(xs[max(0, i - 13):i]) 
            line.set_ydata(ys[max(0, i - 13):i])
            
            ab = AnnotationBbox(imagebox, xy,
                                xycoords='data',
                                pad=0.0,
                                frameon=False)

        
        if i == 0:
            if args.show_histogram:
                query_flat = np.ravel(query_histograms)                
                histo_bar = ax_opti.bar(np.arange(len(query_flat)), query_flat)
            img_artist = ax_inflight.imshow(pic[:,:,0])
        else:
            img_artist.set_data(pic[:,:,0])
            if args.use_sift: sift_drone_artist.remove()
            if args.use_particle_filter: particle_plot.remove()
            if args.use_normal:
                drone_artist.remove()
                
                #ebars[0].remove()
                #for line in ebars[1]:
                #    line.remove()
                #for line in ebars[2]:
                #    line.remove()
            if args.filter: filtered_drone_artist.remove()

            if args.show_histogram:
                query_flat = np.ravel(query_histograms)
                for rect, h in zip(histo_bar, query_flat):
                    rect.set_height(h)

        if args.use_particle_filter:
            particle_plot = ax.scatter(plt_xs, plt_ys)
        if args.use_normal:
            drone_artist = ax.add_artist(ab)
            # Plot particle positions
            #ax.add_artist(particle_plot)

            #ebars = ax.errorbar(xy[0], xy[1], xerr=err_x, yerr=err_y, ecolor='b')
        if args.filter: filtered_drone_artist = ax.add_artist(filtered_ab)
        if args.use_sift: sift_drone_artist = ax.add_artist(sift_ab)

        plt.pause(1e-10)

        # Particle filter
        if args.use_particle_filter:
            ws, w_sum = pf.get_weights(p, xy, dt, i)
            new_p = pf.resample_wheel(p, ws, N)

        
        i += 1
        
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

