from __future__ import division

import cv2
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from collections import Counter
from scipy.spatial import distance
import texton_helpers
import subprocess 
import shlex
import heatmap
import time

from scipy import spatial

# TODO:

# - HEATMAP Display query image in a better way (don't use GIMP
# but cut it out on the fly)
# 
# Look at different distance measurements (Guido doesn't expect too
# good results here)

# Don't cut out only straight but also use rotations, and circles, etc.

# Idea: For an input image, calculate the histogram of clusters, i.e.
# how often does each texton from the dictionary (i.e. the example
# textons or texton cluster centers) occur.

# Afterwards, we do the same for a part of the 'map' and use that for
# a fast pre-localization algorithm

# To do this, we compare the query image with different views of the
# original image (for example with a sliding window)

# Ideally, it can also determine if we are currently between two
# clusters


def tf(histograms):

    return np.sum(histograms, axis=0)

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

    return k_means, predictions, centers


def cluster_textons(textons, classifier):
    
    "This function clusters textons by means of a given classifier"

    classes = classifier.predict(np.float32(textons))
    return classes


def match_histograms(query_histogram, location_histogram):

    """
    Match query histogram with location histogram and return the
    distance. To do this, it needs a distance measurement.
    """
    
    # TODO: I could use the distance function as a parameter

    dist = np.linalg.norm(query_histogram - location_histogram)
    #dist = spatial.distance.cosine(query_histogram, location_histogram)
    #cv2.compareHist(query_histogram, location_histogram, 0)

    return dist
    

def display_textons(textons, input_is_1D=False):

    """
    This function displays the input textons 
    """

    if input_is_1D:
        
        l = len(textons[0])
        s = np.sqrt(l)
        w = int(s) 

        textons = textons.reshape(-1, w, w)

    plt.figure(1) # Create figure

    for i, texton in enumerate(textons):

        plt.subplot(np.ceil(s), np.ceil(s), i + 1) 
        plt.imshow(texton, 
                   cmap = cm.Greys_r, 
                   interpolation="nearest")
    
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


def get_training_histograms(classifier, training_image, n_clusters=20):

    """
    Split the input image into patches and calculate the histogram for each patch
    """

    h, w = training_image.shape

    num_patches_h = 2
    num_patches_w = 3

    # window_offset = (h / 2, w / 2)
    window_offset = None
    
    patches = texton_helpers.sliding_window(training_image, (h / num_patches_h, w / num_patches_w), window_offset, True)

    histograms = []

    # TODO: Could be done faster with map (or other higher-order functions)?!
    for patch in patches:
        
        # Extract textons for each patch
        patch_histogram = img_to_texton_histogram(patch, 
                                                  classifier, 
                                                  max_textons, 
                                                  n_clusters)
        histograms.append(patch_histogram)
        
    weights = 1 / tf(histograms)

    return np.array(histograms), patches, weights


def sliding_window_match(query_image,
                         location_image,
                         max_textons=None,
                         n_clusters=20,
                         SHOW_GRAPHS=True):
    
    """
    This function is the core function of this approach. It matches
    the query image with the location image at different patches. TO
    do this, it calculates the histogram of both images.
    """

    # I assume that I extract all textons from the training image and
    # get the histogram for the different patches off-line. This
    # should increase the speed of the algorithm.
    
    # Extract patches of the training image
    training_textons = extract_textons(location_image, max_textons)

    # Apply K-Means on the training image
    classifier, training_clusters, centers = train_and_cluster_textons(textons=training_textons, 
                                                                       n_clusters=n_clusters)

    # Display dictionary of textons
    if SHOW_GRAPHS:
        display_textons(np.int32(centers), input_is_1D=True)

    # Get histogram of the textons of patches of the training image
    
    training_histograms, patches, weights = get_training_histograms(classifier, training_image, n_clusters)

    query_histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights)
    
    if SHOW_GRAPHS:
        display_histogram(query_histogram)
        
        for patch in patches:

            plt.imshow(patch, 
                   cmap = cm.Greys_r)
            plt.show()

    
    # Perform classification by comparing the histogram of the
    # training and the query image (e.g. with nearest neighbors)
   
    # The output of the classification should be the position of the
    # query image in the localization image.
    
    # TODO: match_histogram should be a helper function for a function
    # that generates sliding windows and matches against them

    distances = []

    for training_histogram in training_histograms:
        dist = match_histograms(query_histogram, weights * training_histogram)
        distances.append(dist)
        

    return distances, patches

def display_query_and_location(query_image, location_image):

    fig = plt.figure()

    fig_1 = fig.add_subplot(2,1,1)
    fig_1.imshow(query_image, 'Greys')

    fig_2 = fig.add_subplot(2,1,2)
    fig_2.imshow(patches[patch_pos], 'Greys')

    plt.show()

def color_map(location_img_c, num_patches_h, num_patches_w, matches, filename='heatmap.png', correct_answer=0):

    h, w, c = location_img_c.shape

    centers = [(0, 0)] * (num_patches_w * num_patches_h)

    step_size_w = (w / num_patches_w)
    step_size_h = (h / num_patches_h)

    step_center_w0 = step_size_w / 2
    step_center_w = step_center_w0
    step_center_h = step_size_h / 2


    i = 0
    for ph in xrange(num_patches_h):
        for pw in xrange(num_patches_w):

            centers[i] = (int(step_center_w), int(step_center_h))            
            step_center_w += step_size_w
            i += 1
        
        step_center_w = step_center_w0
        step_center_h += step_size_h
        
    
    heatmap.heatmap(location_img_c, centers, matches, filename, 'Shepards')


    h_map = plt.imread(filename, 1)

    alpha = 0.9
    beta = 1.0 - alpha

    overlayed = cv2.addWeighted(h_map, alpha,
                                location_img_c, beta,
                                1.0)

    c_x, c_y = centers[correct_answer]

    overlayed = cv2.circle(overlayed, 
                           (c_x, c_y), 
                           40, 
                           (128, 128, 128),
                           -1)

    print centers[correct_answer]

#    plt.imshow(img_masked)
    plt.imshow(overlayed)
    plt.show()

    return centers
        

        
if __name__ == "__main__":

    num_patches_h = 2
    num_patches_w = 3

    SHOW_GRAPHS = True

    # Path of the map (training image)
    #training_image_path = "maze17.jpg"
    #training_image_path = "guy.jpg"
    #training_image_path =  "/home/pold87/Desktop/Bilder/P1080334.JPG"

    # Path of the query image (part of the map)
    #query_image_path = "guynotidealmatch.png"
    #query_image_path = "guybottomright.png"
    #query_image_path = "maze17topleft.png"
    #query_image_path = "entirerightnotperfect.jpg"
    #query_image_path = "righttopnotperfect.jpg"
    #query_image_path = "topmiddlenotperfect.jpg"
    
    image_patches = [("maze17.jpg", "maze17topleft.png", 0),
                     ("maze17.jpg", "entirerightnotperfect.jpg", 2),
                     ("maze17.jpg", "righttopnotperfect.jpg", 2),
                     ("maze17.jpg", "topmiddlenotperfect.jpg", 1),
                     ("guy.jpg", "guynotidealmatch.png", 0),
                     ("guy.jpg", "guybottomright.png", 5)]

    for i, (training_image_path, query_image_path, correct_answer) in enumerate(image_patches):
                     
        training_image = cv2.imread(training_image_path, 0)
        training_img_c = cv2.imread(training_image_path, 1)

        #query_image_path = "tlpic.jpg"
        query_image = cv2.imread(query_image_path, 0)

        # Max textons that are extracted from the input image
        max_textons = 5000
        #max_textons = None

        # Number of clusters (i.e. texton cluster centers)
        n_clusters = 40

        distances, patches = sliding_window_match(query_image=query_image,
                                                  location_image=training_image,
                                                  max_textons=max_textons,
                                                  n_clusters=n_clusters,
                                                  SHOW_GRAPHS=False)

        patch_pos = np.argmin(distances)
        patch_all_pos = np.argsort(distances)

        print "Query image is: ", query_image_path 
        print "Correct Answer is:", correct_answer
        print "Best match", patch_pos
        print "All matches (best matches first)", patch_all_pos
        print ""

        #display_query_and_location(query_image, training_image)
        color_map(training_img_c, num_patches_h, num_patches_w, patch_all_pos, str(i) + '.png', correct_answer)
