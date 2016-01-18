#!/usr/bin/env python

import numpy as np
import pandas as pd
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
from matplotlib._png import read_png
import argparse
from multiprocessing import Process, Value
from sklearn.externals import joblib
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import pickle
import thread
import time
import threading
from treXton import img_to_texton_histogram, RGB2Opponent, imread_opponent
from treXtonConfig import parser

pic = imread_opponent("img/0.png")

for channel in range(3):
    
    mymean = np.mean(np.ravel(pic[:, :, channel]))
    mystdv = np.std(np.ravel(pic[:, :, channel]))

    print("Mymean", mymean)
    print("Mystdv", mystdv)    

    pic[:, :, channel] = pic[:, :, channel] - mymean
    pic[:, :, channel] = pic[:, :, channel] / mystdv


('Mymean', 218.8809752929688)
('Mystdv', 17.88053141401106)
('Mymean', -10.347855143229165)
('Mystdv', 8.9478371853075043)
('Mymean', -28.948378971354153)
('Mystdv', 14.720635663243229)
