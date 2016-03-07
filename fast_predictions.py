from __future__ import division

import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib._png import read_png
import cv2
from treXton import img_to_texton_histogram, RGB2Opponent
import treXton
from matplotlib.offsetbox import OffsetImage
from treXtonConfig import parser
import xgboost as xgb

def prediction_variance(model, X):
    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(X)[0])

    return np.var(preds)

def main(args):

    cap = cv2.VideoCapture(args.dev)

    kmeans = []

    for channel in range(3):
    
        kmean = joblib.load('classifiers/kmeans' + str(channel) + '.pkl')
        kmeans.append(kmean)

    if args.do_separate:
        # Load classifier
        clf_x = joblib.load('classifiers/clf_x.pkl')
        clf_y = joblib.load('classifiers/clf_y.pkl') 

    else:
        clf = joblib.load('classifiers/clf_multi.pkl')

    # Load tfidf
    tfidf = joblib.load('classifiers/tfidf.pkl') 

    # Feature importances
    #for a in zip(range(150), clf_x.feature_importances_):
    #    print a
    #print clf_y.feature_importances_

    fp = open("predictions_cross.csv", "w")

    for i in range(args.num_test_pics):

        query_file = args.test_imgs_path + str(i) + ".png"
        query_image = treXton.imread_opponent(query_file)

        if args.local_standardize:

            mymean, mystdv = cv2.meanStdDev(query_image)
            mymean = mymean.reshape(-1)
            mystdv = mystdv.reshape(-1)

            query_image = (query_image - mymean) / mystdv      

        # Get texton histogram of picture
        query_histograms = np.zeros((args.channels, args.num_textons))

        for channel in range(args.channels):
            kmean = kmeans[channel]        
            histogram = img_to_texton_histogram(query_image[:, :, channel], 
                                                  kmean,
                                                  args.max_textons,
                                                  args.num_textons,
                                                  1, args, channel)
            query_histograms[channel] = histogram
            
        query_histograms = query_histograms.reshape(1, args.num_textons * args.channels)  

        if args.tfidf:
            query_histograms = tfidf.transform(query_histograms).todense()
            query_histograms = np.ravel(query_histograms)

            query_histograms = query_histograms.reshape(1, args.num_textons * args.channels)  

        if args.use_xgboost:
            dtest = xgb.DMatrix(query_histograms)
            pred_x = clf_x.predict(dtest)[0]
            pred_y = clf_y.predict(dtest)[0]
        elif not(args.do_separate):
            print "Not separate"
            pred = clf.predict(query_histograms)[0]
            pred_x = pred[0]
            pred_y = pred[1]
        else:                
            pred_x = clf_x.predict(query_histograms)[0]
            pred_y = clf_y.predict(query_histograms)[0]
        
        if args.prediction_variance:
            pred_var_x = prediction_variance(clf_x, query_histograms)
            pred_var_y = prediction_variance(clf_y, query_histograms)
            fp.write("%f,%f,%f,%f\n" % (pred_x, pred_y, pred_var_x, pred_var_y)) 
        else:
            fp.write("%f,%f\n" % (pred_x, pred_y)) 

    fp.close()

if __name__ == "__main__":

    args = parser.parse_args()    
    main(args)
