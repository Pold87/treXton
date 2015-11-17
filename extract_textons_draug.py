from __future__ import division

import cv2
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
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
#import emd
from math import log, sqrt
from scipy import spatial
import glob
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_draug_pics", type=int, help="The amount of draug pictures to use", default=100)
parser.add_argument("-t", "--num_test_pics", type=int, help="The amount of test images to use", default=50)
parser.add_argument("-d", "--dir", default="/home/pold/Documents/draug/", help="Path to draug directory")
parser.add_argument("-tp", "--test_imgs_path", default="/home/pold/Documents/imgs_first_flight/", help="Path to test images")
parser.add_argument("-g", "--show_graphs", help="Show graphs of textons", action="store_true")
parser.add_argument("-ttr", "--test_on_trainset", help="Test on trainset (calculate training error)", action="store_false")
parser.add_argument("-tte", "--test_on_testset", help="Test on testset (calculate error)", action="store_false")
parser.add_argument("-nt", "--num_textons", help="Size of texton dictionary", type=int, default=100)
parser.add_argument("-mt", "--max_textons", help="Maximum amount of textons per image", type=int, default=500)
parser.add_argument("-o", "--use_optitrack", help="Use optitrack", action="store_true")
args = parser.parse_args()

# Settings
base_dir = args.dir
genimgs_path = base_dir + "genimgs/"
testimgs_path = args.test_imgs_path
coordinates = pd.read_csv(base_dir + "targets.csv")
num_draug_pics = args.num_draug_pics # Number of pictures to include from draug
num_test_pics = args.num_test_pics # Number of pictures to test
test_on_trainset = args.test_on_trainset # Calculate error on trainset
test_on_testset = args.test_on_testset # Calculate predictons on testset (real world data)
use_optitrack = args.use_optitrack # Calculate errors on testset using Optitrack (real world data)
SHOW_GRAPHS = args.show_graphs

# Idea: For an input image, calculate the histogram of clusters, i.e.
# how often does each texton from the dictionary (i.e. the example
# textons or texton cluster centers) occur.

# Afterwards, we do the same for a part of the 'map' and use that for
# a fast pre-localization algorithm

# To do this, we compare the query image with different views of the
# original image (for example with a sliding window)

# Ideally, it can also determine if we are currently between two
# clusters.

def xlog(xi, yi):
    if xi == 0 or yi == 0:
        return 0
    else:
        return xi*log(float(xi)/float(yi),2)

def KLD(x,y):
    """ Kullback Leibler divergence """
    return sum([xlog(xi, yi) for xi, yi in zip(x, y)])

def JSD(p, q):
    """ Jensen Shannon divergence """
    p = np.array(p)
    q = np.array(q)
    return sqrt(0.5* KLD(p, 0.5*(p + q)) + 0.5 * KLD(q, 0.5*(p + q)))

def Jeffrey(p, q):
    """ Jeffreys divergence """
    j = 0
    for a, b in zip(p, q):
        if a == 0 or b == 0:
            pass
        else:
            j += (a-b)*(log(a)-log(b))
    return j


def extract_textons(img, max_textons=None, texton_size=5):

    """
    This function extract textons from an image. If max_textons is set
    to None, all textons are extracted, otherwise random sampling is
    used.
    """

    patches = image.extract_patches_2d(img, 
                                       (texton_size, texton_size),
                                       max_textons)

    # Flatten 2D array
    patches = patches.reshape(-1, texton_size ** 2)

    return patches


def extract_textons_from_path(path, max_textons=100, texton_size=5):

    """
    This function extract textons from an image. If max_textons is set
    to None, all textons are extracted, otherwise random sampling is
    used.
    """

#    genimgs = glob.glob(genimgs_path + '*.png')

    all_patches = []

    for pic_num in range(num_draug_pics):

        genimg_file = genimgs_path + str(pic_num) + ".png"
        
        genimg = cv2.imread(genimg_file, 0)

        print(genimg_file)

        patches = image.extract_patches_2d(genimg, 
                                           (texton_size, texton_size),
                                           max_textons)

        # Flatten 2D array
        patches = patches.reshape(-1, texton_size ** 2)

        all_patches.extend(patches)

    return all_patches

    
def train_and_cluster_textons(textons, n_clusters=25):

    """
    Returns a classifier, learned from the orignal image, and the
    predictions for the classes of the input textons and the texton
    centers.

    """

    # TODO: Look at different parameters of n_init
    k_means = KMeans(n_clusters=n_clusters, n_init=10)

    # Predicted classes for the input textons
    predictions = k_means.fit_predict(np.float32(textons))

    # Texton class centers
    centers = k_means.cluster_centers_

    if SHOW_GRAPHS:
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


def img_to_texton_histogram(img, classifier, max_textons, n_clusters, weights=None):

    # Extract all textons of the query image
    textons = extract_textons(img, max_textons)

    # Get classes of textons of the query image
    clusters = cluster_textons(textons, classifier)

    # Get the frequency of each texton class of the query image
    histogram = np.bincount(clusters,
                            minlength=n_clusters) # minlength guarantees that missing clusters are set to 0 

    if weights is not None:
        histogram = histogram * weights

    return histogram


def train_classifier_draug(path,
                           max_textons=None,
                           n_clusters=20):

    # Extract patches of the training image
    training_textons = extract_textons_from_path(path, max_textons)        

    # Apply k-Means on the training image
    classifier, training_clusters, centers = train_and_cluster_textons(textons=training_textons, 
                                                                       n_clusters=n_clusters)
    histograms = []

    y_top_left = []
    y_bottom_right = []

    rf_top_left = RandomForestRegressor(500)
    rf_bottom_right = RandomForestRegressor(500)

    weights = 1

#    genimgs = glob.glob(genimgs_path + "*.png")

    for i in range(num_draug_pics):

        genimg = genimgs_path + str(i) + ".png"
        
        query_image = cv2.imread(genimg, 0)

        top_left_x = coordinates.ix[i, "x"]
        top_left_y = coordinates.ix[i, "y"]
        y_top_left.append((top_left_x, top_left_y))

        query_histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights)
            
        histograms.append(query_histogram)


    rf_top_left.fit(histograms, y_top_left)

    return rf_top_left, histograms, y_top_left, classifier, weights



def display_query_and_location(query_image, location_image):

    fig = plt.figure()

    fig_1 = fig.add_subplot(2,1,1)
    fig_1.imshow(query_image, 'Greys')

    fig_2 = fig.add_subplot(2,1,2)
    fig_2.imshow(patches[patch_pos], 'Greys')

    plt.show()


def create_random_patch(img,
                        min_window_width=100,
                        min_window_height=100,
                        max_window_width=300,
                        max_window_height=300):

    """

    Extracts a random path from a given image and returns the image and
    the coordinates in the original image.

    """

    h = img.shape[0]
    w = img.shape[1]

    # print "imageshape", img.shape

    pos_y = np.random.randint(0, h - min_window_height)
    pos_x = np.random.randint(0, w - min_window_width)


    # print "pos_y, pos_x", pos_y,pos_x
    
    
    height = np.random.randint(min_window_height, min(h - pos_y, max_window_height))
    width = np.random.randint(min_window_width, min(w - pos_x, max_window_width))

    # print 'height,widht', height, width

    top_left = (pos_x, pos_y)
    bottom_right = (pos_x + width, pos_y + height)
    
    patch = img[pos_y : pos_y + height, pos_x : pos_x + width]
    
    return patch, top_left, bottom_right


def main_draug():
    
    max_textons = args.max_textons
    n_clusters = args.num_textons
    
    path = genimgs_path

    rf_top_left,  histograms, y_top_left, classifier, weights = train_classifier_draug(
        path=path,
        max_textons=max_textons,
        n_clusters=n_clusters)

    
    if test_on_trainset:

        errors_x = []
        errors_y = []
        
        for i in range(num_draug_pics):

            query_file = genimgs_path + str(i) + ".png"
            query_image = cv2.imread(query_file, 0)

            # previously: histograms[patch]
            histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights)

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

    if test_on_testset:

        predictions = []

        offset = 50 # Discard the first pictures
        for i in range(offset, offset + num_test_pics):

            query_file = testimgs_path + str(i) + ".jpg"
            query_image = cv2.imread(query_file, 0)

            histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights)

            if SHOW_GRAPHS:
                print("hello")
                display_histogram(histogram)

            pred_top_left = rf_top_left.predict([histogram])
            print "pred is", pred_top_left

            predictions.append(pred_top_left[0])

        np.save("predictions", np.array(predictions))

        if use_optitrack: 
            pass
            


                
if __name__ == "__main__":

        main_draug()

