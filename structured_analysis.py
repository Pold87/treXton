import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import scipy
import cv2
import warnings
from matplotlib.cbook import get_sample_data
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
    AnnotationBbox
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neighbors import LSHForest, DistanceMetric
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
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
from treXtonConfig import parser
import pickle

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

args = parser.parse_args()
mymap = args.mymap

space = {
    # Number of channels (Grayscale vs color)
    'channels_spec': hp.choice('channel_spec', [
        {
            'channels': 1,
            'standardization': hp.choice('standardization_c1',
                                         ['local_standardize', 'none'])
            
        },
        {
            'channels': 3,
            'standardization': hp.choice('standardization_c3',
                                         ['local_standardize', 'color_standardize', 'none'])

        }
        ]),
    # General treXton-based parameters        
    'num_textons': hp.quniform('num_textons', 10, 100, 1),
    'max_textons': hp.quniform('max_textons', 100, 2000, 100),
    'texton_size': hp.quniform('texton_size', 2, 10, 1),
    'tfidf': hp.choice('tfidf', [True, False]),

    # Try out different classifiers (model selection)
    'classifier_type': hp.choice('classifier_type', [
        {
            'type': 'knn',
        },
#        {
#            'type': 'svm',
#            'C': hp.lognormal('svm_C', 0, 1),
#            'kernel': hp.choice('svm_kernel', [
#                {'ktype': 'linear'},
#                {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
#                ]),
#        },
        {
            'type': 'randomforest',
            'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
            'max_depth': hp.choice('dtree_max_depth',
                    [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
            'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
        },
    ])}

trials = Trials()

def objective(p):

    if p['classifier_type']['type'] == 'knn':
        clf_x = KNeighborsRegressor()
        clf_y = KNeighborsRegressor()
    elif p['classifier_type']['type'] == 'randomforest':
        clf_x = RandomForestRegressor(max_depth=p['classifier_type']['max_depth'],
                                    min_samples_split=p['classifier_type']['min_samples_split'])
        clf_y = RandomForestRegressor(max_depth=p['classifier_type']['max_depth'],
                                    min_samples_split=p['classifier_type']['min_samples_split'])

        
    pickle.dump(clf_x, open("hyperopt_clf_x.p", "wb"))
    pickle.dump(clf_y, open("hyperopt_clf_y.p", "wb"))
        
    # Simple command
    subprocess.call(['treXton.py', '-c',
                     '--channels', str(p['channels_spec']['channels']),                     
                     '--num_textons', str(int(p['num_textons'])),
                     '--max_textons', str(int(p['max_textons'])),
                     '--texton_size', str(int(p['texton_size'])),
                     '--load_clf_settings'
                     ])
    subprocess.call(['fast_test.py',
                     '--channels', str(p['channels_spec']['channels']),                                          
                     '--num_textons', str(int(p['num_textons'])),
                     '--max_textons', str(int(p['max_textons'])),
                     '--texton_size', str(int(p['texton_size'])),
                     '--load_clf_settings'
                      ])
    mse, mse_x, mse_y = np.load("all_errors.npy")
    print(p)
    print("ME", np.sqrt(mse), "ME_x", np.sqrt(mse_x), "ME_y", np.sqrt(mse_y))
    print("")
    return {'loss' : mse,
            'status' : STATUS_OK}

trials = Trials()

best = fmin(fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=70,
    trials=trials)

print best
