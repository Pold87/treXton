from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import scipy
from matplotlib._png import read_png
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
import treXton
import relocalize
import configargparse
import treXtonConfig
import seaborn as sns
from treXtonConfig import parser
import time
import particle_filter as pf
import xgboost as xgb
import bayes

def main(args):

    labels = np.load("labels.npy")
    histograms = np.load("histograms_logos.npy")

    cap = cv2.VideoCapture(args.dev)

    kmeans = []

    for channel in range(3):
    
        kmean = joblib.load('classifiers/kmeans' + str(channel) + '.pkl')
        kmeans.append(kmean)

        
    minidrone = read_png("img/minidrone.png")
    background_map = plt.imread("../image_recorder/general_images/mosaic.png")
    imagebox = OffsetImage(minidrone, zoom=1)
    ax = plt.gca()
    ax.imshow(background_map)
    k = 0

    while True:

        if k != 0:
            drone_artist.remove()
        
        distances = []

        ret, pic_bgr = cap.read()
        pic = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2RGB)
        pic = RGB2Opponent(pic)

        cv2.imshow("Capture", pic_bgr)

        gray = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        thresholding = False

        if thresholding:
        
            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1

            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0

            markers = cv2.watershed(pic_bgr, markers)
            pic_bgr[markers == -1] = [255,0,0]
        
        #cv2.imshow("Threshold", pic_bgr)
        
        if args.local_standardize:
            for channel in range(args.channels):
                mymean = np.mean(np.ravel(pic[:, :, channel]))
                mystdv = np.std(np.ravel(pic[:, :, channel]))

                pic[:, :, channel] = pic[:, :, channel] - mymean
                pic[:, :, channel] = pic[:, :, channel] / mystdv
        

        # Get texton histogram of picture
        query_histograms = np.zeros((args.channels, args.num_textons))

        for channel in range(args.channels):
            histogram = img_to_texton_histogram(pic[:, :, channel],
                                                kmeans[channel],
                                                args.max_textons,
                                                args.num_textons,
                                                1,
                                                args,
                                                channel)
            query_histograms[channel] = histogram

        query_histograms = query_histograms.reshape(1, -1)

        for hist in histograms:
            hist = hist.reshape(1, -1)
            dist = np.linalg.norm(hist[0] - query_histograms[0])
            distances.append(dist)

        distances = np.array(distances)
        min_dist = distances.min()
        arg_min = distances.argmin()

        #print(min_dist)

        sorted_dists = np.sort(distances)

        #print(sorted_dists[:2])
        #print("")

        sorted_labels = [x for (y,x) in sorted(zip(distances, labels))]

        clf = joblib.load("classifiers/logo_clf.pkl")

        pred = clf.predict(query_histograms.reshape(1, -1))
        probs = clf.predict_proba(query_histograms.reshape(1, -1))

        signs = ['linux', 'camel', 'firefox']
        
        for i in zip(signs, probs[0]):
            print i
        print("")

        xy = bayes.maximum_a_posteriori(*list(probs[0]))

        #if any(i >= 0.75 for i in probs[0]):
        #    print("pred is", pred[0])
        #else:
        #    print("pred is", "background")

        xy = (xy[0], xy[1])

        print(xy)
        
        ab = AnnotationBbox(imagebox, xy,
                            xycoords='data',
                            pad=0.0,
                            frameon=False)

        drone_artist = ax.add_artist(ab)
        k += 1
        plt.pause(1e-10)        
        cv2.waitKey(1)

        
        #print(probs)
        
#
#if min_dist < 200 and sorted_labels[0] == sorted_labels[1] == sorted_labels[2] == sorted_labels[3] == sorted_labels[4]:
#    pass
#    #print(sorted_dists[0])
#    #print(labels[arg_min])
#
#else:
#    print("Background")
#    
        

    


if __name__ == "__main__":

    args = parser.parse_args()    
    main(args)
