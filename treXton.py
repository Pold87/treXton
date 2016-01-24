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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.linalg import get_blas_funcs
from scipy.linalg import blas as FB
from sklearn.utils.extmath import fast_dot

tfidf = TfidfTransformer()

def mydist(x, y):

    x_norm = x / np.sum(x)
    y_norm = y / np.sum(y)

    return np.sum((x_norm - y_norm) ** 2 / (x_norm + y_norm + 1e-8)) / 2


def RGB2Opponent(img):

    A = np.array([[0.06, 0.63, 0.27],
                   [0.30, 0.04, -0.35],
                   [0.34, -0.60, 0.17]]).astype(np.float32)
    img = img.astype(np.float32)

    #prod = pbcvt.dot(img.reshape(480 * 640, 3), A.T).reshape(480, 640, 3)
    prod = np.dot(img.reshape(480 * 640, 3), A.T).reshape(480, 640, 3)
    
    return prod



def calc_sim(t1, t2, s = 10):

    t1 = t1.reshape(-1)
    t2 = t2.reshape(-1)    
    
    dist = np.linalg.norm(t1 - t2)
    sim = np.exp((- dist) / s ** 2)
    
    return sim

def extract_one_texton(img, x, y, texton_size_x, texton_size_y):
    texton = img[y:y + texton_size_x, x:x + texton_size_y, 0]
    return texton

def my_extract_patches(img, texton_size, max_textons):

    h = img.shape[0]
    w = img.shape[1]    

    texton_positions = np.zeros((max_textons, 2))

    patches = np.zeros((max_textons,
                        texton_size[0], texton_size[1]))
    
    for i in range(max_textons):

        x = np.random.randint(w - texton_size[0] - 1)
        y = np.random.randint(h - texton_size[1] - 1)

        texton_positions[i] = np.array([x, y])

        extracted_texton = extract_one_texton(img, x, y,
                                        texton_size[0], texton_size[1])

        patches[i] = extracted_texton

    return patches, texton_positions

def extract_textons(img, max_textons, args, real_max_textons, channel):

    """
    This function extract textons from an image. If max_textons is set
    to None, all textons are extracted, otherwise random sampling is
    used.
    """

    if args.use_dipoles:
        patches, texton_positions = my_extract_patches(img, 
                                                       (args.texton_size, args.texton_size),
                                                       real_max_textons)
        
    else:
        patches = image.extract_patches_2d(img, 
                                           (args.texton_size, args.texton_size),
                                           real_max_textons)
        texton_positions = []

    # Flatten 2D array
    patches = patches.reshape(-1, args.texton_size ** 2)

    
    new_patches = []

    new_zero = 0
    if args.standardize:
        
        mean, stdv = np.load("mean_stdv_" + str(channel) + ".npy")
        new_zero = - mean / stdv

    #if args.local_standardize:
        
    #    mymean = np.mean(np.ravel(img))
    #    mystdv = np.std(np.ravel(img))

    #    new_zero = - mymean / mystdv        

    counter = 0
    if args.resample_textons:
        for patch in patches:
            #if not all(patch == patch[0]):

            if not all(patch == new_zero) or not all(patch == 0):
                new_patches.append(patch)
                counter += 1
            if counter == max_textons: break

        if len(new_patches) == 0:
            new_patches.append(patches[0])
    else:
        new_patches = patches

    
        
    return new_patches, texton_positions

def extract_textons_from_path(path, max_textons=100, channel=0):

    """
    This function extract textons from an image. If max_textons is set
    to None, all textons are extracted, otherwise random sampling is
    used.
    """

#    genimgs = glob.glob(genimgs_path + '*.png')

    all_patches = []

    # For standarization
    img_means_per_channel = []
    img_vars_per_channel = []

    start = 0
    stop = 80
    step = 5

    for pic_num in range(start, stop, step):


        if args.use_draug_folder:
            genimg_file = path + str(pic_num) + "_0.png"
        else:
            genimg_file = path + str(pic_num) + ".png"
        
        genimg = imread_opponent(genimg_file)
        genimg = genimg[:, :, channel]

        img_means_per_channel.append(genimg.mean())
        img_vars_per_channel.append(genimg.var())

    
    if channel == 0:
        np.save("imgs_vars.npy", img_vars_per_channel)
        np.save("imgs_vars_per_channel.npy", img_vars_per_channel)        
        np.save("imgs_means_per_channel.npy", img_means_per_channel)
        
    else:
        img_vars = np.load("imgs_vars.npy")

    if args.standardize:
        mean_imgs = np.mean(img_means_per_channel)
        stdv_imgs = np.sqrt(np.mean(img_vars_per_channel))

        print("mean:", mean_imgs)
        print("stdv:", stdv_imgs)
    
        vals = np.array([mean_imgs, stdv_imgs])
        np.save("mean_stdv_" + str(channel) + ".npy", vals)

    k = 0
    for pic_num in range(start, stop, step):

        if args.use_draug_folder:
            genimg_file = path + str(pic_num) + "_0.png"
        else:
            genimg_file = path + str(pic_num) + ".png"

        genimg = imread_opponent(genimg_file)
        genimg = genimg[:, :, channel]

        if args.color_standardize:
            if channel == 0:
                mymean = np.mean(np.ravel(genimg))
                mystdv = np.std(np.ravel(genimg))

                genimg = genimg - mymean
                genimg = genimg / mystdv

            else:
                print("img_vars[k]", img_vars[k])
                genimg = genimg / np.sqrt(img_vars[k])

        if args.local_standardize:

            mymean = np.mean(np.ravel(genimg))
            mystdv = np.std(np.ravel(genimg))

            genimg = genimg - mymean
            genimg = genimg / mystdv


        if args.histogram_standardize:

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            genimg = clahe.apply(genimg)

        if args.use_dipoles:
            delta_x = cv2.Sobel(genimg, cv2.CV_64F, 1, 0, ksize=3)
            invar = delta_x / genimg

            # TODO: this function should get the entire image as inpit

        if args.standardize:
            mean, stdv = np.load("mean_stdv_" + str(channel) + ".npy")
            genimg = genimg - mean
            genimg = genimg / stdv


        patches = image.extract_patches_2d(genimg, 
                                           (args.texton_size, args.texton_size),
                                           max_textons)

        # Flatten 2D array
        patches = patches.reshape(-1, args.texton_size ** 2)

        all_patches.extend(patches)
        k += 1

    return all_patches

    
def train_and_cluster_textons(textons, n_clusters=25, channel=0):

    """
    Returns a classifier, learned from the orignal image, and the
    predictions for the classes of the input textons and the texton
    centers.

    """

    # TODO: Look at different parameters of n_init
    k_means = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=-1)

    # Predicted classes for the input textons
    predictions = k_means.fit_predict(np.float32(textons))

    # Texton class centers
    centers = k_means.cluster_centers_

    if args.show_graphs:
        display_textons(centers, channel)

    return k_means, predictions, centers


def cluster_textons(textons, classifier):
    
    "This function clusters textons by means of a given classifier"

    classes = classifier.predict(np.float32(textons))
    return classes


def match_histograms(query_histogram, location_histogram, weights=None):

    """
    Match query histogram with location histogram and return the
    distance. To do this, it needs a distance measurement.
    """
    
    # TODO: I could use the distance function as a parameter


    #dist = np.linalg.norm(query_histogram - location_histogram)
    #dist = spatial.distance.cosine(query_histogram, location_histogram)

    dist = cv2.compareHist(np.float32(query_histogram), np.float32(location_histogram), 4)
    #dist = JSD(np.float32(query_histogram), np.float32(location_histogram))
    #_, dist = stats.ks_2samp(query_histogram, location_histogram)
    #dist = -dist
 #   dist = cv2.EMD(np.float32(query_histogram), np.float32(location_histogram), 3)
    #dist=  stats.entropy(np.float32(query_histogram), np.float32(location_histogram))

    #f = np.float64([1] * len(query_histogram))
    #s = np.float64([1] * len(query_histogram))
    
    #dist = emd.emd(np.float64(query_histogram), np.float64(location_histogram), f, s)
    
    return dist
    

def display_textons(textons, channel=0, input_is_1D=False, save=True):

    """
    This function displays the input textons 
    """

    l = len(textons[0])
    s = np.sqrt(l)
    w = int(s) 

    textons = textons.reshape(-1, w, w)

    plt.figure(1) # Create figure

    d = np.ceil(np.sqrt(len(textons)))
    
    for i, texton in enumerate(textons):

        plt.subplot(d, d, i + 1) 
        plt.imshow(texton, 
                   cmap = cm.Greys_r, 
                   interpolation="nearest")
    
    plt.savefig("extracted_textons_" + str(channel) + ".png")
    #plt.show()



def display_histogram(histogram):

    plt.bar(np.arange(len(histogram)), histogram)
    plt.show()

def do_dipole_transformations(img):

    # Calculate gradients
    sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3, scale=4)
    sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3, scale=4)

    # Color invariant

    mean_x, std_x = cv2.meanStdDev(sobelx)
    mean_y, std_y = cv2.meanStdDev(sobely)

    invar_x = sobelx / std_x.T
    invar_y = sobely / std_y.T

    invar_grad = np.sqrt(invar_x ** 2 + invar_y ** 2)
    invar_direction = np.arctan(invar_y / (invar_x + 1e-8))

    return invar_grad, invar_direction
    
    
def get_histogram_dipole(img, clusters, texton_positions, args):

    invar_grad, invar_direction = do_dipole_transformations(img)


    edge_strengths = np.sqrt(invar_grad[0] ** 2 + invar_grad[1] ** 2 + invar_grad[2] ** 2)

    edge_strengths = edge_strengths / np.sum(edge_strengths)

    invar_direction = invar_direction * edge_strengths
    
    s = 1

    histogram = np.zeros((args.num_textons, 4))

    for i, (x, y) in enumerate(texton_positions):

        
        
        intens = invar_direction[y + args.texton_size, x + args.texton_size, 0] #.reshape(-1)
        intens_inverse = - intens
        by = invar_direction[y + args.texton_size, x + args.texton_size, 1] #.reshape(-1)
        gr = invar_direction[y + args.texton_size, x + args.texton_size, 2] #.reshape(-1)

        dist1 = np.linalg.norm(intens - by)
        dist2 = np.linalg.norm(by - intens_inverse)
        dist3 = np.linalg.norm(intens - gr)
        dist4 = np.linalg.norm(gr - intens_inverse)

        sim1 = np.exp((- dist1) / s ** 2)
        sim2 = np.exp((- dist2) / s ** 2)
        sim3 = np.exp((- dist3) / s ** 2)
        sim4 = np.exp((- dist4) / s ** 2)

#        print(sim1)
#        print(sim2)
#        print(sim3)
#        print(sim4)
#
#        print("")
#
        histogram[clusters[i]] += np.array([sim1, sim2, sim3, sim4])


        
    histogram = histogram.reshape(-1)

    return histogram


def img_to_texton_histogram(img, classifier, max_textons, n_clusters, weights, args, channel):

    # Extract all textons of the query image
    if args.resample_textons:
        total_textons = int(max_textons * 1.5)
    else:
        total_textons = int(max_textons)

    start_extraction = time.time()
    textons, texton_positions = extract_textons(img, max_textons, args, total_textons, channel)
    end_extraction = time.time()
    if args.measure_time:        
        print("extraction", end_extraction - start_extraction)        

    # Get classes of textons of the query image
    start_clustering = time.time()    
    clusters = cluster_textons(textons, classifier)
    end_clustering = time.time()    
    if args.measure_time:        
        print("clustering", end_clustering - start_clustering)        


    if args.use_dipoles:
        histogram = get_histogram_dipole(img, clusters, texton_positions, args)
    else:
    # Get the frequency of each texton class of the query image
        histogram = np.bincount(clusters,
                                minlength=n_clusters) # minlength guarantees that missing clusters are set to 0
    

    #weights = len(textons) / args.max_textons
                            
    #if weights is not None:
    #    histogram = histogram / weights

    return histogram

def imread_opponent(path):

    # Read as RGB
    img = plt.imread(path, 1)

    # Convert to opponent space
    #start = time.time()
    img = RGB2Opponent(img)
    #end = time.time()
    #print("Time convert", end - start)

    return img

def imread_opponent_gray(path):

    return imread_opponent(path)[:, :, 0]

def train_regression_draug(path,
                           max_textons=None,
                           n_clusters=20,
                           args=None):

    # Settings
    base_dir = args.dir
    #genimgs_path = base_dir + "genimgs/"
    genimgs_path = base_dir
    testimgs_path = args.test_imgs_path
    #coordinates = pd.read_csv(base_dir + "targets_gtl.csv")
    coordinates_gtl = pd.read_csv(args.ground_truth_labeler)
   # coordinates_draug = pd.read_csv("../draug/targets.csv")

    classifiers = []
    if args.clustering:

        for channel in range(args.channels):

            # Extract patches of the training image
            training_textons = extract_textons_from_path(path, max_textons, channel)

            # Apply k-Means on the training image
            classifier, training_clusters, centers = train_and_cluster_textons(textons=training_textons, 
                                                                           n_clusters=n_clusters, channel=channel)

            classifiers.append(classifier)
            
            joblib.dump(classifier, 'classifiers/kmeans' + str(channel) + '.pkl')
            

    else:

        for channel in range(args.channels):

            # Load classifier from file
            classifier = joblib.load('classifiers/kmeans' + str(channel) + '.pkl')
            classifiers.append(classifier)


    histograms = []
    y_top_left = []
    y_bottom_right = []
    num_matches = []    

    

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import sklearn.linear_model as lm
#    rf_top_left = RandomForestRegressor(500, n_jobs=-1)
#    rf_top_left = KNeighborsRegressor(p=1)


    arguments = {
        'booster': 'gbtree',
        'eval_metric': 'rmse',
        'eta': 0.01,
        'max_depth': 15,
        'n_estimators':500,
        'nthread': 4,
        'subsample': 0.95,
        'min_child_weight': 1,
        'gamma': 0.01,
        'colsample_bytree': 0.95,
        'colsample_bylevel': 0.95,
        'nthread': 4,
        'silent': False
    }
    
    weights = 1

#    genimgs = glob.glob(genimgs_path + "*.png")

    picturenumbers = np.random.randint(0, 220, 100)
    picturenumbers = range(0, args.num_draug_pics, 1)

    if args.load_histograms:
         histograms = np.load("histograms.npy")
         y_top_left = np.load("y_top_left.npy")
         y_bottom_right = np.load("y_bottom_right.npy")                           
    else:

        if args.use_draug_folder:
            picturevariants = 15
        else:
            picturevariants = 1

        for i in picturenumbers:

            for j in range(picturevariants):

                if picturevariants == 1:
                    genimg = genimgs_path + str(i) + ".png"
                else:
                    genimg = genimgs_path + str(i) + "_" + str(j) + ".png"


                query_image = imread_opponent(genimg)


                if args.local_standardize:
                    for channel in range(args.channels):
                        mymean = np.mean(np.ravel(query_image[:, :, channel]))
                        mystdv = np.std(np.ravel(query_image[:, :, channel]))

                        query_image[:, :, channel] = query_image[:, :, channel] - mymean
                        query_image[:, :, channel] = query_image[:, :, channel] / mystdv

                if args.color_standardize:
                    mymean = np.mean(np.ravel(query_image[:, :, 0]))
                    mystdv = np.std(np.ravel(query_image[:, :, 0]))

                    query_image[:, :, 0] = query_image[:, :, 0] - mymean
                    query_image[:, :, 0] = query_image[:, :, 0] / mystdv                
                    query_image[:, :, 1] = query_image[:, :, 1] / mystdv
                    query_image[:, :, 2] = query_image[:, :, 2] / mystdv            

                if args.histogram_standardize:

                    # create a CLAHE object (Arguments are optional).
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    for channel in range(args.channels):
                        query_image[:, :, channel] = clahe.apply(query_image[:, :, channel])

                                        
                if args.standardize:
                    for channel in range(args.channels):
                        mean, stdv = np.load("mean_stdv_" + str(channel) + ".npy")
                        query_image[:, :, channel] = query_image[:, :, channel] - mean
                        query_image[:, :, channel] = query_image[:, :, channel] / stdv

                top_left_x = coordinates_gtl.ix[i, "x"]
                top_left_y = coordinates_gtl.ix[i, "y"]
                matches = coordinates_gtl.ix[i, "matches"]
                num_matches.append(matches)

                if args.do_separate:
                    y_top_left.append(top_left_x)
                    y_bottom_right.append(top_left_y)
                else:
                    y_top_left.append((top_left_x, top_left_y))


                query_histograms = []
                for channel in range(args.channels):
                    classifier = classifiers[channel]
                    if args.use_dipoles:
                        query_histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights, args, channel)
                    else:
                        query_histogram = img_to_texton_histogram(query_image[:, :, channel], classifier, max_textons, n_clusters, weights, args, channel)
                    query_histograms.append(query_histogram)

                query_histograms = np.ravel(query_histograms)

                histograms.append(query_histograms)
        np.save("histograms.npy", np.array(histograms))
        np.save("y_top_left.npy", np.array(y_top_left))
        np.save("y_bottom_right.npy", np.array(y_bottom_right))
        np.save("num_matches.npy", np.array(num_matches))                        
                

    if args.tfidf:
        histograms = tfidf.fit_transform(histograms).todense()
        joblib.dump(tfidf, 'classifiers/tfidf.pkl')

    if args.do_separate:
        if args.load_clf_settings:
            clf_x_coord = pickle.load(open("hyperopt_clf_x.p", "rb" ))
            clf_y_coord = pickle.load(open("hyperopt_clf_y.p", "rb" ))
        else:
            pass
            #K = chi2_kernel(histograms, gamma=.5)                
            #dist = DistanceMetric.get_metric(mydist)        
            #clf_x_coord = lm.TheilSenRegressor()
            #clf_y_coord = lm.TheilSenRegressor()

            #clf_x_coord = xgb.XGBRegressor(**arguments)
            #clf_y_coord = xgb.XGBRegressor(**arguments)
            #clf_x_coord = svm.LinearSVR(epsilon=0)
            #clf_y_coord = svm.LinearSVR(epsilon=0)
            #clf_x_coord = RandomForestRegressor(50, n_jobs=-1)
            #clf_y_coord = RandomForestRegressor(50, n_jobs=-1)
            #clf_x_coord = lm.BayesianRidge(normalize=True)
            #clf_y_coord = lm.BayesianRidge(normalize=True)
            #clf_x_coord = AdaBoostRegressor()
            #clf_y_coord = AdaBoostRegressor()            
            #clf_x_coord = GradientBoostingRegressor()
            #clf_y_coord = GradientBoostingRegressor()
            #clf_x_coord = GaussianProcessRegressor()
            #clf_y_coord = GaussianProcessRegressor()
            clf_x_coord = KNeighborsRegressor(weights='distance', metric=mydist)
            clf_y_coord = KNeighborsRegressor(weights='distance', metric=mydist)
            #clf_x_coord = KNeighborsRegressor(3, weights='distance')
            #clf_y_coord = KNeighborsRegressor(3, weights='distance')
            
            #clf_x_coord = LinearRegression()
            #clf_y_coord = LinearRegression()                
            #clf_x_coord = MLPRegressor()
            #clf_y_coord = MLPRegressor()


    else:
        clf0 = RandomForestRegressor(500, n_jobs=-1)
        clf1 = RandomForestRegressor(500, n_jobs=-1)
        #clf0 = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
        #clf1 = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
        #clf0 = KNeighborsRegressor(algorithm='kd_tree', weights='distance', n_neighbors=11)
        #clf0 = KNeighborsRegressor(weights='uniform', n_neighbors=7, p=3)
        #clf1 = KNeighborsRegressor(algorithm='kd_tree', weights='distance', n_neighbors=9)
        clfs = [clf0, clf1]
        
        

    if args.do_separate:
        print("Fitting")

        if args.use_xgboost:
            num_round = 500
            #evallist  = [(dtest,'eval'), (dtrain,'train')]
            dtrain_x = xgb.DMatrix(histograms,
                                 label=y_top_left,
                                 weight=np.array(num_matches))
            dtrain_y = xgb.DMatrix(histograms,
                                 label=y_bottom_right,
                                 weight=np.array(num_matches))
            
            bst_x = xgb.train(arguments,
                            dtrain_x,
                            num_round)
            
            bst_y = xgb.train(arguments,
                            dtrain_y,
                            num_round)
            clf = bst_x
            joblib.dump(bst_x, 'classifiers/clf_x.pkl')
            joblib.dump(bst_y, 'classifiers/clf_y.pkl')             

            
        else:

            if args.sample_weight:
                print("Using sample weight")
                clf_x_coord.fit(histograms, y_top_left, sample_weight=np.array(num_matches))
                clf_y_coord.fit(histograms, y_bottom_right, sample_weight=np.array(num_matches))

            else:
                clf_x_coord.fit(histograms, y_top_left)
                print(clf_x_coord.predict(histograms))
                clf_y_coord.fit(histograms, y_bottom_right)
            print("Finished Fitting")
            clf = clf_x_coord
            joblib.dump(clf_x_coord, 'classifiers/clf_x.pkl')
            joblib.dump(clf_y_coord, 'classifiers/clf_y.pkl') 
        
    else:
        for j, clf in enumerate(clfs):
            clf.fit(histograms, y_top_left)
            joblib.dump(clf, "classifiers/clf" + str(j) + ".pkl") 

    return clf, histograms, y_top_left, classifier, weights



def display_query_and_location(query_image, location_image):

    fig = plt.figure()

    fig_1 = fig.add_subplot(2,1,1)
    fig_1.imshow(query_image, 'Greys')

    fig_2 = fig.add_subplot(2,1,2)
    fig_2.imshow(patches[patch_pos], 'Greys')

    plt.show()


def main_draug(args):

    # Settings
    
    base_dir = args.dir
    genimgs_path = base_dir + "genimgs/"
    testimgs_path = args.test_imgs_path
#    coordinates = pd.read_csv(base_dir + "targets.csv")
    coordinates = pd.read_csv(args.ground_truth_labeler)
    
    max_textons = args.max_textons
    n_clusters = args.num_textons
    
    path = base_dir

    rf_top_left,  histograms, y_top_left, classifier, weights = train_regression_draug(
        path=path,
        max_textons=max_textons,
        n_clusters=n_clusters,
        args=args)


    if args.test_on_trainset:

        errors_x = []
        errors_y = []
        
        for i in range(args.num_draug_pics):

            query_file = genimgs_path + str(i) + ".png"
            query_image = imread_opponent_gray(query_file)

            # previously: histograms[patch]
            histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights, args)

            top_left_x = coordinates.ix[i, "x"]
            top_left_y = coordinates.ix[i, "y"]
            
            pred_top_left = rf_top_left.predict([histogram])
            print ("pred is", pred_top_left)
            print ("real values", (top_left_x, top_left_y))

            diff_x = abs(pred_top_left[0][0] - top_left_x)
            diff_y = abs(pred_top_left[0][1] - top_left_y)
            
            print("diff x", diff_x)
            print("diff y", diff_y)

            errors_x.append(diff_x)
            errors_y.append(diff_y)

        print("Mean error x", np.mean(errors_x))
        print("Mean error y", np.mean(errors_y))

    if args.test_on_validset:

        errors_x = []
        errors_y = []
        
        for i in range(args.start_valid, args.start_valid + args.num_valid_pics):

            query_file = genimgs_path + str(i) + ".png"
            query_image = imread_opponent_gray(query_file)

            # previously: histograms[patch]
            histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights, args)

            top_left_x = coordinates.ix[i, "x"]
            top_left_y = coordinates.ix[i, "y"]
            
            pred_top_left = rf_top_left.predict([histogram])
            print("pred is", pred_top_left)
            print("real values", (top_left_x, top_left_y))

            diff_x = abs(pred_top_left[0][0] - top_left_x)
            diff_y = abs(pred_top_left[0][1] - top_left_y)
            
            print("diff x", diff_x)
            print("diff y", diff_y)

            errors_x.append(diff_x)
            errors_y.append(diff_y)

        print("Mean error x (valid)", np.mean(errors_x))
        print("Mean error y (valid)", np.mean(errors_y))



    if args.test_on_testset:

        predictions = []

        offset = args.start_pic_num # Discard the first pictures
        for i in range(offset, offset + args.num_test_pics):

            query_file = testimgs_path + str(i) + ".jpg"
            query_image = imread_opponent_gray(query_file)

            histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights, args)

            if args.show_graphs:
                display_histogram(histogram)

            pred_top_left = rf_top_left.predict([histogram])
            print("pred is", pred_top_left)

            predictions.append(pred_top_left[0])

        np.save("predictions", np.array(predictions))

        if args.use_optitrack: 
            pass

if __name__ == "__main__":

    args = parser.parse_args()
    
    main_draug(args)

