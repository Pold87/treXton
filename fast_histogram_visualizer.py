import treXton
import sys
import numpy as np
import cv2
from sklearn.externals import joblib
import argparse
import treXtonConfig


def display_textons(args):

    mean, stdv = np.load("mean_stdv.npy")

    print(mean, stdv)
    
    img = cv2.imread(args.image, 0)

    print(args.image)


    img = img -  mean
    img = img / stdv


    print(img[0:30])
    
    
    # Load classifier from file
    classifier = joblib.load('classifiers/kmeans.pkl') 
    
    hist = treXton.img_to_texton_histogram(img, classifier, args.max_textons, args.num_textons, 1, args)
    for i, t in enumerate(hist):
        print i, t


if __name__ == "__main__":

    args = treXtonConfig.parse_me()

    print(args)

    display_textons(args)
    
