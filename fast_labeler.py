import numpy as np
import treXton
import cv2
from sklearn.externals import joblib
from treXtonConfig import parser

def main(args):

    basedir = "../image_recorder/playing_mat"
    num_histograms = 100
    channel = 0
    classifier = joblib.load('classifiers/kmeans' + str(channel) + '.pkl')
    max_textons = 300
    n_clusters = 33
    weights = 1

    texton_hists = []

    for i in range(num_histograms):
        img_dir = basedir + "/" + str(i) + ".png"
        img = treXton.imread_opponent(img_dir)

        texton_hist = treXton.img_to_texton_histogram(img[:, :, 0], classifier, max_textons, n_clusters, weights, args, channel)
        texton_hists.append(texton_hist)

    np.savetxt("texton_histograms.csv", texton_hists, delimiter=",", fmt='%d') 


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)
