from __future__ import division

import cv2
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neighbors import LSHForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from collections import Counter
from scipy.spatial import distance
import texton_helpers
import subprocess 
import shlex
import heatmap
import time
import sys
import math
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

tfidf = TfidfTransformer()

def RGB2Opponent(img):

    A = np.array([[0.06, 0.63, 0.27],
                   [0.30, 0.04, -0.35],
                   [0.34, -0.60, 0.17]])

    return np.dot(img, A.T)

def extract_textons(img, max_textons, args, real_max_textons, channel):

    """
    This function extract textons from an image. If max_textons is set
    to None, all textons are extracted, otherwise random sampling is
    used.
    """

    patches = image.extract_patches_2d(img, 
                                       (args.texton_size, args.texton_size),
                                       real_max_textons)

    # Flatten 2D array
    patches = patches.reshape(-1, args.texton_size ** 2)

    
    new_patches = []

    new_zero = 0
    if args.standardize:
        
        mean, stdv = np.load("mean_stdv_" + str(channel) + ".npy")
        new_zero = - mean / stdv

    counter = 0
    for patch in patches:
        #if not all(patch == patch[0]):
        
        if not all(patch == new_zero) or not all(patch == 0):
            new_patches.append(patch)
            counter += 1
        if counter == max_textons: break

    if len(new_patches) == 0:
        new_patches.append(patches[0])

    print("Channel", channel, "Size", len(new_patches))

    return new_patches


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
    stop = 90
    step = 1

    for pic_num in range(start, stop, step):


        if args.use_draug_folder:
            genimg_file = path + str(pic_num) + "_0.png"
        else:
            genimg_file = path + str(pic_num) + ".png"
        print(genimg_file)
        
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
                genimg = genimg / img_vars[k]
                print(genimg)

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

    
def train_and_cluster_textons(textons, n_clusters=25):

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
        display_textons(centers)

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
    

def display_textons(textons, input_is_1D=False, save=True):

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
    
    plt.savefig("extract_textons.png")
    plt.show()



def display_histogram(histogram):

    plt.bar(np.arange(len(histogram)), histogram)
    plt.show()


def img_to_texton_histogram(img, classifier, max_textons, n_clusters, weights, args, channel):

    # Extract all textons of the query image
    textons = extract_textons(img, max_textons, args, 1000, channel)

    # Get classes of textons of the query image
    clusters = cluster_textons(textons, classifier)

    # Get the frequency of each texton class of the query image
    histogram = np.bincount(clusters,
                            minlength=n_clusters) # minlength guarantees that missing clusters are set to 0

#    histogram[73] = 0

#    print("Amount textons:", len(textons) - twentysixer)

    #weights = len(textons) / args.max_textons
                            
    #if weights is not None:
    #    histogram = histogram / weights

    return histogram

def imread_opponent(path):

    # Read as RGB
    img = plt.imread(path, 1)

    # Convert to opponent space
    img = RGB2Opponent(img)

    return img

def imread_opponent_gray(path):

    return imread_opponent(path)[:, :, 0]


def train_classifier_draug(path,
                           max_textons=None,
                           n_clusters=20,
                           args=None):

    # Settings
    base_dir = args.dir
    #genimgs_path = base_dir + "genimgs/"
    genimgs_path = base_dir
    testimgs_path = args.test_imgs_path
    #coordinates = pd.read_csv(base_dir + "targets_gtl.csv")
    coordinates_gtl = pd.read_csv("boodschappen.csv")
   # coordinates_draug = pd.read_csv("../draug/targets.csv")

    classifiers = []
    if args.clustering:

        for channel in range(3):

            # Extract patches of the training image
            training_textons = extract_textons_from_path(path, max_textons, channel)

            # Apply k-Means on the training image
            classifier, training_clusters, centers = train_and_cluster_textons(textons=training_textons, 
                                                                           n_clusters=n_clusters)

            classifiers.append(classifier)
            
            joblib.dump(classifier, 'classifiers/kmeans' + str(channel) + '.pkl')
            

    else:

        for channel in range(3):

            # Load classifier from file
            classifier = joblib.load('classifiers/kmeans' + str(channel) + '.pkl')
            classifiers.append(classifier)


    histograms = []
    y_top_left = []
    y_bottom_right = []

    

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression 
#    rf_top_left = RandomForestRegressor(500, n_jobs=-1)
#    rf_top_left = KNeighborsRegressor(p=1)


    arguments = {'max_depth': 15,
                 'learning_rate': 0.01,
                 'n_estimators':1200,
                 'subsample': 0.85,
                 'min_child_weight': 6,
                 'gamma': 2,
                 'colsample_bytree': 0.75,
                 'colsample_bylevel': 0.9,
                 'silent': True}
    
    if args.do_separate:
        clf_x_coord = xgb.XGBRegressor(**arguments)
        clf_y_coord = xgb.XGBRegressor(**arguments)
        #clf_x_coord = svm.LinearSVR(epsilon=0)
        #clf_y_coord = svm.LinearSVR(epsilon=0)
        #clf_x_coord = RandomForestRegressor(500, n_jobs=-1)
        #clf_y_coord = RandomForestRegressor(500, n_jobs=-1)
        
    
    else:
        clf0 = RandomForestRegressor(500)
        #clf1 = RandomForestRegressor(100)
        #clf0 = KNeighborsRegressor(algorithm='kd_tree', weights='distance', n_neighbors=11)
        #clf0 = KNeighborsRegressor(weights='uniform', n_neighbors=7, p=3)
        clf1 = KNeighborsRegressor(algorithm='kd_tree', weights='distance', n_neighbors=9)
        clfs = [clf0, clf1]
        

    weights = 1

#    genimgs = glob.glob(genimgs_path + "*.png")

    picturenumbers = np.random.randint(0, 220, 100)
    picturenumbers = range(0, 95, 1)

    if args.use_draug_folder:
        picturevariants = 10
    else:
        picturevariants = 1
        
    for i in picturenumbers:

        for j in range(picturevariants):

            if picturevariants == 1:
                genimg = genimgs_path + str(i) + ".png"
            else:
                genimg = genimgs_path + str(i) + "_" + str(j) + ".png"


            query_image = imread_opponent(genimg)

            mymean = np.mean(np.ravel(query_image[:, :, 0]))
            mystdv = np.std(np.ravel(query_image[:, :, 0]))

            if args.color_standardize:
                query_image[:, :, 0] = query_image[:, :, 0] - mymean
                query_image[:, :, 0] = query_image[:, :, 0] / mystdv                
                query_image[:, :, 1] = query_image[:, :, 1] / mystdv
                query_image[:, :, 2] = query_image[:, :, 2] / mystdv            
            
            if args.standardize:
                for channel in range(3):
                    mean, stdv = np.load("mean_stdv_" + str(channel) + ".npy")
                    query_image[:, :, channel] = query_image[:, :, channel] - mean
                    query_image[:, :, channel] = query_image[:, :, channel] / stdv
                
            top_left_x = coordinates_gtl.ix[i, "x"]
            top_left_y = coordinates_gtl.ix[i, "y"]

            if args.do_separate:
                y_top_left.append(top_left_x)
                y_bottom_right.append(top_left_y)
            else:
                y_top_left.append((top_left_x, top_left_y))


            query_histograms = []
            for channel in range(3):
                classifier = classifiers[channel]
                query_histogram = img_to_texton_histogram(query_image[:, :, channel], classifier, max_textons, n_clusters, weights, args, channel)
                query_histograms.append(query_histogram)

            query_histograms = np.ravel(query_histograms)

            histograms.append(query_histograms)

    # Get histograms and targets from draug

    use_draug = False

    if use_draug:

        for i in range(0, 4800, 1):

            mydraugpath = "../draug/genimgs/"
            genimg = mydraugpath + str(i) + ".png"

            query_image = imread_opponent_gray(genimg)

            if args.standardize:
                query_image = query_image - mean
                query_image = query_image / stdv

            #cv2.imwrite(str(i) + "_normalized.png", query_image)

            top_left_x = coordinates_draug.ix[i, "x"]
            top_left_y = coordinates_draug.ix[i, "y"]

            if args.do_separate:
                y_top_left.append(top_left_x)
                y_bottom_right.append(top_left_y)
            else:
                y_top_left.append((top_left_x, top_left_y))


            query_histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights, args)

            histograms.append(query_histogram)

    if args.tfidf:
        histograms = tfidf.fit_transform(histograms).todense()
        joblib.dump(tfidf, 'classifiers/tfidf.pkl') 

    if args.do_separate:
        print("Fitting")
        clf_x_coord.fit(histograms, y_top_left)
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
    coordinates = pd.read_csv("boodschappen.csv")

    # Idea: For an input image, calculate the histogram of clusters, i.e.
    # how often does each texton from the dictionary (i.e. the example
    # textons or texton cluster centers) occur.

    # Afterwards, we do the same for a part of the 'map' and use that for
    # a fast pre-localization algorithm

    # To do this, we compare the query image with different views of the
    # original image (for example with a sliding window)

    # Ideally, it can also determine if we are currently between two
    # clusters.

    
    max_textons = args.max_textons
    n_clusters = args.num_textons
    
    path = base_dir

    rf_top_left,  histograms, y_top_left, classifier, weights = train_classifier_draug(
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
            print "pred is", pred_top_left
            print "real values", (top_left_x, top_left_y)

            diff_x = abs(pred_top_left[0][0] - top_left_x)
            diff_y = abs(pred_top_left[0][1] - top_left_y)
            
            print "diff x", diff_x
            print "diff y", diff_y

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
            print "pred is", pred_top_left
            print "real values", (top_left_x, top_left_y)

            diff_x = abs(pred_top_left[0][0] - top_left_x)
            diff_y = abs(pred_top_left[0][1] - top_left_y)
            
            print "diff x", diff_x
            print "diff y", diff_y

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
            print "pred is", pred_top_left

            predictions.append(pred_top_left[0])

        np.save("predictions", np.array(predictions))

        if args.use_optitrack: 
            pass
            


                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-nd", "--num_draug_pics", type=int, help="The amount of draug pictures to use", default=300)
    parser.add_argument("-nv", "--num_valid_pics", type=int, help="The amount of valid pictures to use", default=49)
    parser.add_argument("-sv", "--start_valid", type=int, help="Filenumber of the first valid picture", default=950)
    parser.add_argument("-t", "--num_test_pics", type=int, help="The amount of test images to use", default=40)
    parser.add_argument("-d", "--dir", default="../draug/genimgs_folder/", help="Path to draug directory")    
    parser.add_argument("-tp", "--test_imgs_path", default="imgs_straight/", help="Path to test images")
    parser.add_argument("-s", "--start_pic_num", type=int, default=20, help="Discard the first pictures (offset)")
    parser.add_argument("-g", "--show_graphs", help="Show graphs of textons", action="store_true")
    parser.add_argument("-ttr", "--test_on_trainset", help="Test on trainset (calculate training error)", action="store_false")
    parser.add_argument("-tte", "--test_on_testset", help="Test on testset (calculate error)", action="store_false")
    parser.add_argument("-tv", "--test_on_validset", help="Test on validset (calculate valid error)", action="store_false")
    parser.add_argument("-nt", "--num_textons", help="Size of texton dictionary", type=int, default=30)
    parser.add_argument("-mt", "--max_textons", help="Maximum amount of textons per image", type=int, default=700)
    parser.add_argument("-o", "--use_optitrack", help="Use optitrack", action="store_true")
    parser.add_argument("-ts", "--texton_size", help="Size of the textons", type=int, default=5)
    parser.add_argument("-c", "--clustering", default=False, help="Do clustering or load clusters from file", action="store_true")
    parser.add_argument("-tfidf", "--tfidf", default=True, help="Perform tfidf", action="store_false")
    parser.add_argument("-std", "--standardize", default=True, help="Perform standarization", action="store_false")
    parser.add_argument("-ds", "--do_separate", default=True, help="Use two classifiers (x and y)", action="store_false")
    parser.add_argument("-udf", "--use_draug_folder", default=False, help="Use picture enhanced by draug (folder)", action="store_true")
    parser.add_argument("-cs", "--color_standardize", default=False, help="Standardize channel 2 and 3 by dividing them by channel 1", action="store_true")
    args = parser.parse_args()
    
    main_draug(args)

