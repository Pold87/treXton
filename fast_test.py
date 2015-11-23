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


parser = argparse.ArgumentParser()
parser.add_argument("-tp", "--test_imgs_path", default="/home/pold/Documents/datasets/mat/", help="Path to test images")
parser.add_argument("-m", "--mymap", default="../draug/img/bestnewmat.png", help="Path to the mat image")
parser.add_argument("-p", "--predictions", default="predictions.npy", help="Path to the predictions of extract_textons_draug.py")
parser.add_argument("-c", "--camera", default=False, help="Use camera for testing", action="store_true")
parser.add_argument("-mo", "--mode", default=0, help="Use the camera (0), test on train pictures (1), test on test pictures (2)", type=int)
parser.add_argument("-s", "--start_pic", default=950, help="Starting picture (offset)", type=int)
parser.add_argument("-n", "--num_pictures", default=500, help="Amount of pictures for testing", type=int)
parser.add_argument("-ts", "--texton_size", help="Size of the textons", type=int, default=7)
parser.add_argument("-nt", "--num_textons", help="Size of texton dictionary", type=int, default=100)
parser.add_argument("-mt", "--max_textons", help="Maximum amount of textons per image", type=int, default=500)
parser.add_argument("-tfidf", "--tfidf", default=True, help="Perform tfidf", action="store_false")
parser.add_argument("-std", "--standardize", default=True, help="Perform standarization", action="store_false")
parser.add_argument("-ds", "--do_separate", default=True, help="Use two classifiers (x and y)", action="store_false")
parser.add_argument("-f", "--filter", default=True, help="Use Kalman filter for filtering", action="store_false")
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

def main():

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


        
    path = args.test_imgs_path
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

    for i in labels.index:

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

        preds = []
        if args.do_separate:
            pred_x = clf_x.predict([histogram])
            pred_y = clf_y.predict([histogram])
            pred  = np.array([(pred_x[0], pred_y[0])])
        else:
            for clf in clfs:
                pred = clf.predict([histogram])
                preds.append(pred)

            pred = np.mean(preds, axis=0)

        if args.filter:
            my_filter.update(pred.T)
            filtered_pred = (my_filter.x[0][0], my_filter.x[2][0])
            my_filter.predict()


        #print("Ground truth (x, y)", xs_opti[i], ys_opti[i])
        #print("Prediction (x, y)", pred[0][0], pred[0][1])

        if test_on_the_fly:
            if args.do_separate:
                xy = (pred_x[0], pred_y[0])
            else:
                xy = (pred[0][0], pred[0][1])

        else:
            xy = (xs[i], ys[i])


        ground_truth =  (labels.x[i], labels.y[i])
        diff =  np.subtract(ground_truth, xy)
        abs_diff = np.fabs(diff)
        errors_x.append(abs_diff[0])
        errors_y.append(abs_diff[1])
        error = np.linalg.norm(abs_diff)
        errors.append(error)
                
          
    print("errors", np.mean(errors))
    print("errors x", np.mean(errors_x))
    print("errors y", np.mean(errors_y))
         

        

if __name__ == "__main__":
    main()
