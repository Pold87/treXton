#!/usr/bin/env python

from __future__ import division

import cv2
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neighbors import LSHForest, DistanceMetric
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from collections import Counter
from scipy.spatial import distance
import texton_helpers
import subprocess 
import shlex
import heatmap
import time
import sys
import math
import warnings
from scipy import stats
from math import log, sqrt
from scipy import spatial
import glob
import os
import argparse
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
import xgboost as xgb
import configargparse
from treXtonConfig import parser
from sklearn.metrics.pairwise import chi2_kernel
from sknn.backend import lasagne
from sklearn.neural_network import MLPRegressor
from sknn.mlp import Regressor, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.linalg import get_blas_funcs
from scipy.linalg import blas as FB
from sklearn.utils.extmath import fast_dot
import pbcvt
import treXton

symbols = ['vw',
           'mcdonalds',
           'paulaner',
           'starbucks',
           'dominos',
           'paulaner_blurry',
           'background',
           'shell',
           'spotify',
           'chrome',
           'ubuntu',
           'camel']

symbols = ['linux',
           'firefox',
           'logitech',
           'camel']

def train_classifier_draug(path,
                           max_textons=None,
                           n_clusters=20,
                           args=None):

    classifiers = []


    for channel in range(args.channels):

        # Load classifier from file
        classifier = joblib.load('classifiers/kmeans' + str(channel) + '.pkl')
        classifiers.append(classifier)


    histograms = []
    labels = []

    base_dir = "/home/pold87/Documents/Internship/orthomap/"

    for symbol in symbols:

        for i in range(args.max_imgs_per_class):

            genimg_path = base_dir + symbol + '/' + str(i) + '.png'
            if os.path.exists(genimg_path):
            
                query_image = treXton.imread_opponent(genimg_path)
                labels.append(symbol)
                query_histograms = []


                if args.local_standardize:
                    for channel in range(args.channels):
                        mymean = np.mean(np.ravel(query_image[:, :, channel]))
                        mystdv = np.std(np.ravel(query_image[:, :, channel]))

                        query_image[:, :, channel] = query_image[:, :, channel] - mymean
                        query_image[:, :, channel] = query_image[:, :, channel] / mystdv



                for channel in range(args.channels):

                    classifier = classifiers[channel]

                    if args.use_dipoles:
                        query_histogram = treXton.img_to_texton_histogram(query_image,
                                                                  classifier,
                                                                  max_textons,
                                                                  n_clusters,
                                                                  1,
                                                                  args,
                                                                  channel)
                    else:
                        query_histogram = treXton.img_to_texton_histogram(query_image[:, :, channel],
                                                                  classifier,
                                                                  max_textons,
                                                                  n_clusters,
                                                                  1,
                                                                args,
                                                                  channel)
                    query_histograms.append(query_histogram)

                query_histograms = np.ravel(query_histograms)

                histograms.append(query_histograms)

        np.save("histograms_logos.npy", np.array(histograms))
        np.save("labels.npy", np.array(labels))
        clf = RandomForestClassifier(n_estimators=100,
                                     max_depth=15)
        clf.fit(np.array(histograms), np.array(labels))
        joblib.dump(clf, 'classifiers/logo_clf.pkl')


def main(args):

    train_classifier_draug(args.logos_path,
                           args.max_textons,
                           args.num_textons,
                           args=args)


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    main(args)

    

    
