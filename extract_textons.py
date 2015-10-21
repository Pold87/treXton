from __future__ import division

import cv2
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
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



def extract_textons_color(img, max_textons=None, texton_size=5):

    """
    This function extract textons from an image. If max_textons is set
    to None, all textons are extracted, otherwise random sampling is
    used.
    """

    patches = image.extract_patches_2d(img, 
                                       (texton_size, texton_size),
                                       max_textons)

    blues_p = patches[:, :, :, 0]
    greens_p = patches[:, :, :, 1]
    reds_p = patches[:, :, :, 2]

    blues_p_flat = blues_p.reshape(-1, texton_size ** 2)
    greens_p_flat = greens_p.reshape(-1, texton_size ** 2)
    reds_p_flat = reds_p.reshape(-1, texton_size ** 2)
    
    all_p_flat = np.array((reds_p_flat, greens_p_flat, blues_p_flat))
    
    return all_p_flat

    
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


def img_to_texton_histogram_color(img,
                                  (classifier_r,
                                  classifier_g,
                                  classifier_b),
                                  max_textons,
                                  n_clusters,
                                  weights=[1, 1, 1]):

    # Extract all textons of the query image
    textons_r, textons_g, textons_b  = extract_textons_color(img, max_textons)

    # Get classes of textons of the query image
    clusters_r = cluster_textons(textons_r, classifier_r)
    clusters_g = cluster_textons(textons_g, classifier_g)
    clusters_b = cluster_textons(textons_b, classifier_b)

    # Get the frequency of each texton class of the query image
    histogram_r = np.bincount(clusters_r,
                            minlength=n_clusters) # minlength guarantees that missing clusters are set to 0 

    histogram_g = np.bincount(clusters_g,
                            minlength=n_clusters) # minlength guarantees that missing clusters are set to 0 

    histogram_b = np.bincount(clusters_b,
                            minlength=n_clusters) # minlength guarantees that missing clusters are set to 0 
    
                            
    if weights is not None:
        histogram_r = histogram_r * weights[0]
        histogram_g = histogram_g * weights[1]
        histogram_b = histogram_b * weights[2]

    return np.concatenate((histogram_r, histogram_g, histogram_b))


def get_training_histograms(classifier, training_image, num_patches_h, num_patches_w, n_clusters=20, max_textons=20):

    """
    Split the input image into patches and calculate the histogram for each patch
    """

    print training_image
    print training_image.shape

    h = training_image.shape[0]
    w = training_image.shape[1]

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
        
    #weights = 1 / np.sum(histograms, axis=0) # 1 divided by term frequency


    # print 'weights', weights
    
    weights = 1  # without weights

    return np.array(histograms), patches, weights




def train_classifier(location_image,
                     max_textons=None,
                     num_patches_h=2,
                     num_patches_w=3,
                     n_clusters=20):


    # Extract patches of the training image
    training_textons = extract_textons(location_image, max_textons)


    # Apply k-Means on the training image
    classifier, training_clusters, centers = train_and_cluster_textons(textons=training_textons, 
                                                                       n_clusters=n_clusters)


    # Get histogram of the textons of patches of the training image

# TODO: seems to be unnecessary
       
#    training_histograms, patches, weights = get_training_histograms(classifier,
#                                                                    location_image,
#                                                                    num_patches_h,
#                                                                    num_patches_w,
#                                                                    n_clusters,
#                                                                    max_textons)


    histograms = []

    y_top_left = []
    y_bottom_right = []

    rf_top_left = RandomForestRegressor(500)
    rf_bottom_right = RandomForestRegressor(500)

    #rf_top_left = ExtraTreesRegressor()
    #rf_bottom_right = ExtraTreesRegressor()

        
    for i in range(10000):

        # Create random patch with assumed position
        query_image, top_left, bottom_right = create_random_patch(location_image)

        y_top_left.append(top_left)
        y_bottom_right.append(bottom_right)
        
        query_histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights)

        histograms.append(query_histogram)

    rf_top_left.fit(histograms, y_top_left)
    rf_bottom_right.fit(histograms, y_bottom_right)

    return rf_top_left, rf_bottom_right, histograms, y_top_left, y_bottom_right, classifier, weights



def train_classifier_color(location_image,
                           max_textons=None,
                           num_patches_h=2,
                           num_patches_w=3,
                           n_clusters=20):
    

    # Extract patches of the training image
    training_textons_r, training_textons_g, training_textons_b = extract_textons_color(location_image, max_textons)


    # Apply k-Means on the training image
    classifier_r, training_clusters_r, centers_r = train_and_cluster_textons(textons=training_textons_r, 
                                                                       n_clusters=n_clusters)

    classifier_g, training_clusters_g, centers_g = train_and_cluster_textons(textons=training_textons_g, 
                                                                       n_clusters=n_clusters)

    classifier_b, training_clusters_b, centers_b = train_and_cluster_textons(textons=training_textons_b, 
                                                                       n_clusters=n_clusters)


    histograms = []
    weights = [1, 1, 1]
    classifiers = [classifier_r, classifier_g, classifier_b]

    y_top_left = []
    y_bottom_right = []

    rf_top_left = RandomForestRegressor(500)
    rf_bottom_right = RandomForestRegressor(500)

    #rf_top_left = ExtraTreesRegressor()
    #rf_bottom_right = ExtraTreesRegressor()

    plt.imshow(location_image)

        
    for i in range(10000):

        # Create random patch with assumed position
        query_image, top_left, bottom_right = create_random_patch_color(location_image)

        y_top_left.append(top_left)
        y_bottom_right.append(bottom_right)
        
        query_histogram = img_to_texton_histogram_color(query_image,
                    classifiers,
                    max_textons,
                    n_clusters,
                    weights)

        histograms.append(query_histogram)

    rf_top_left.fit(histograms, y_top_left)
    rf_bottom_right.fit(histograms, y_bottom_right)

    return rf_top_left, rf_bottom_right, histograms, y_top_left, y_bottom_right, classifiers, weights

    

def sliding_window_match(query_image,
                         location_image,
                         max_textons=None,
                         num_patches_h=2,
                         num_patches_w=3,
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
    
    training_histograms, patches, weights = get_training_histograms(classifier,
                                                                    training_image,
                                                                    num_patches_h,
                                                                    num_patches_w,
                                                                    n_clusters,
                                                                    max_textons)

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
        #dist = match_histograms(query_histogram, weights * training_histogram)
        dist = match_histograms(query_histogram, training_histogram, weights)
        distances.append(dist)
        

    return distances, patches

def display_query_and_location(query_image, location_image):

    fig = plt.figure()

    fig_1 = fig.add_subplot(2,1,1)
    fig_1.imshow(query_image, 'Greys')

    fig_2 = fig.add_subplot(2,1,2)
    fig_2.imshow(patches[patch_pos], 'Greys')

    plt.show()


    
def color_map(location_img_c, num_patches_h, num_patches_w,
              matches, filename='heatmap.png',
              correct_answer=None,
              top_left=None, bottom_right=None):

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

    if correct_answer is not None:
        c_x, c_y = centers[correct_answer]

        overlayed = cv2.circle(overlayed, 
                               (c_x, c_y), 
                               40, 
                               (128, 128, 128),
                               -1)


    if top_left is not None:

        overlayed = cv2.rectangle(overlayed,
                                  top_left,
                                  bottom_right,
                                  (255, 0, 0),
                                  10)
                                  

    fig = plt.figure()
    fig.add_subplot(111)
    
    plt.imshow(overlayed)

    plt.show()
        
    return centers
        

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


def create_random_patch_color(img,
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
    
    patch = img[pos_y : pos_y + height, pos_x : pos_x + width, :]
    
    return patch, top_left, bottom_right
    

def main():
    READ_FROM_DISK = False
    CREATE_ON_THE_FLY = True

    num_patches_h = 6
    num_patches_w = 6

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
    
    image_patches = [#("maze17.jpg", "maze17topleft.png", 0),
                     #("maze17.jpg", "entirerightnotperfect.jpg", 2),
        #("maze17.jpg", "righttopnotperfect.jpg", 2),
        #("maze17.jpg", "righttopnotperfect.jpg", 2),
        #("maze17.jpg", "righttopnotperfect.jpg", 2),
        ("img/background.jpg", "topmiddlenotperfect.jpg", 1),
        ("img/background.jpg", "topmiddlenotperfect.jpg", 1),
        ("img/guy.jpg", "guynotidealmatch.png", 0),
        ("img/guy.jpg", "guybottomright.png", 5)]
        

    for i, (training_image_path, query_image_path, correct_answer) in enumerate(image_patches):

        training_image = cv2.imread(training_image_path, 0)
        training_img_c = cv2.imread(training_image_path, 1)

        #query_image_path = "tlpic.jpg"

        if READ_FROM_DISK:
            query_image = cv2.imread(query_image_path, 0)

        elif CREATE_ON_THE_FLY:
            query_image, top_left, bottom_right = create_random_patch(training_image)

        # Max textons that are extracted from the input image
        max_textons = 1000
        #max_textons = None

        # Number of clusters (i.e. texton cluster centers)
        n_clusters = 50

        distances, patches = sliding_window_match(query_image=query_image,
                                                  location_image=training_image,
                                                  max_textons=max_textons,
                                                  num_patches_h=num_patches_h,
                                                  num_patches_w=num_patches_w,
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

        if READ_FROM_DISK:
            color_map(training_img_c, num_patches_h, num_patches_w, patch_all_pos, str(i) + '.png', correct_answer=correct_answer)


        if CREATE_ON_THE_FLY:

            print "Top left", top_left
            print "Bottom right", bottom_right

            #cv2.imshow('frame', query_image)
            #cv2.waitKey(0)
            
            color_map(training_img_c, num_patches_h, num_patches_w, patch_all_pos, str(i) + '.png', top_left=top_left, bottom_right=bottom_right)


def main2():
    
    READ_FROM_DISK = False
    CREATE_ON_THE_FLY = True

    num_patches_h = 6
    num_patches_w = 6

    SHOW_GRAPHS = True

    training_image_path = 'webcam-take.jpg'

    max_textons = 500
    n_clusters = 40
    
    training_image = cv2.imread(training_image_path, 0)

    rf_top_left, rf_bottom_right, histograms, y_top_left, y_bottom_right, classifier, weights = train_classifier(training_image,
                     max_textons=max_textons,
                     num_patches_h=2,
                     num_patches_w=3,
                     n_clusters=n_clusters)


    test_on_trainset = False
    
    if test_on_trainset:
        for patch in range(30):


            pred_top_left = rf_top_left.predict([histograms[patch]])
            pred_bottom_right = rf_bottom_right.predict([histograms[patch]])
            print "pred is", pred_top_left, pred_bottom_right
            print "real values", y_top_left[patch], y_bottom_right[patch]


            overlayed = cv2.imread(training_image_path, 1)


            overlayed = cv2.rectangle(overlayed,
                                      (y_top_left[patch][0], y_top_left[patch][1]),
                                      (y_bottom_right[patch][0], y_bottom_right[patch][1]),
                                      (255, 0, 0),
                                      10)


            overlayed = cv2.rectangle(overlayed,
                                      (int(pred_top_left[0][0]), int(pred_top_left[0][1])),
                                      (int(pred_bottom_right[0][0]), int(pred_bottom_right[0][1])),
                                      (0, 255, 0),
                                      10)


            plt.imshow(overlayed)
            plt.show()

    else:

        for patch in range(20000):
            # Create random patch with assumed position
            query_image, top_left, bottom_right = create_random_patch(training_image)

            query_histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights)


            pred_top_left = rf_top_left.predict([query_histogram])
            pred_bottom_right = rf_bottom_right.predict([query_histogram])
            print "pred is", pred_top_left, pred_bottom_right
            print "real values", top_left, bottom_right

            overlayed = cv2.imread(training_image_path, 1)


            overlayed = cv2.rectangle(overlayed,
                                      top_left,
                                      bottom_right,
                                      (255, 0, 0),
                                      10)


            overlayed = cv2.rectangle(overlayed,
                                      (int(pred_top_left[0][0]), int(pred_top_left[0][1])),
                                      (int(pred_bottom_right[0][0]), int(pred_bottom_right[0][1])),
                                      (0, 255, 0),
                                      10)


            plt.imshow(overlayed)
            plt.show()


            
def main_video():

    training_image_path = 'webcam-take.jpg'

    max_textons = 1000
    n_clusters = 50
    
    training_image = cv2.imread(training_image_path, 0) # gray scale
    training_image = cv2.imread(training_image_path, 1) # color

    rf_top_left, rf_bottom_right, histograms, y_top_left, y_bottom_right, classifier, weights = train_classifier(training_image,
                     max_textons=max_textons,
                     num_patches_h=2,
                     num_patches_w=3,
                     n_clusters=n_clusters)


    test_on_trainset = False

    video = cv2.VideoCapture(0)
    
    while video.isOpened():

        ret, query_image = video.read()

        query_img = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        
        query_histogram = img_to_texton_histogram(query_image, classifier, max_textons, n_clusters, weights)

        
        pred_top_left = rf_top_left.predict([query_histogram])
        pred_bottom_right = rf_bottom_right.predict([query_histogram])
        print "pred is", pred_top_left, pred_bottom_right

        overlayed = cv2.imread(training_image_path, 1)

        overlayed = cv2.rectangle(overlayed,
                                  (int(pred_top_left[0][0]), int(pred_top_left[0][1])),
                                  (int(pred_bottom_right[0][0]), int(pred_bottom_right[0][1])),
                                  (0, 255, 0),
                                  10)

        # Display the resulting frame
        cv2.imshow('query_image', query_image)
        cv2.imshow('frame', overlayed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    
            
def main_video_color():

    training_image_path = 'webcam-take.jpg'

    max_textons = 1000
    n_clusters = 50
    
    training_image = cv2.imread(training_image_path, 1) # color

    rf_top_left, rf_bottom_right, histograms, y_top_left, y_bottom_right, classifiers, weights = train_classifier_color(training_image,
                     max_textons=max_textons,
                     num_patches_h=2,
                     num_patches_w=3,
                     n_clusters=n_clusters)


    test_on_trainset = False

    video = cv2.VideoCapture(0)
    
    while video.isOpened():

        ret, query_image = video.read()

        query_histogram = img_to_texton_histogram_color(query_image, classifiers, max_textons, n_clusters, weights)


        print query_histogram
        print len(query_histogram)
        
        pred_top_left = rf_top_left.predict([query_histogram])
        pred_bottom_right = rf_bottom_right.predict([query_histogram])
        print "pred is", pred_top_left, pred_bottom_right

        overlayed = cv2.imread(training_image_path, 1)

        overlayed = cv2.rectangle(overlayed,
                                  (int(pred_top_left[0][0]), int(pred_top_left[0][1])),
                                  (int(pred_bottom_right[0][0]), int(pred_bottom_right[0][1])),
                                  (0, 255, 0),
                                  10)

        # Display the resulting frame
        cv2.imshow('query_image', query_image)
        cv2.imshow('frame', overlayed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    



                
                    
        
if __name__ == "__main__":

        main_video_color()

