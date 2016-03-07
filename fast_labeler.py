import numpy as np
import treXton
import cv2
from sklearn.externals import joblib
from treXtonConfig import parser

def mydist(x, y):

    x_norm = x / np.sum(x)
    y_norm = y / np.sum(y)

    return np.sum((x_norm - y_norm) ** 2 / (x_norm + y_norm + 1e-8)) / 2


def standardize(img, channel):
    for channel in range(args.channels):
        mymean = np.mean(np.ravel(img[:, :, channel]))
        mystdv = np.std(np.ravel(img[:, :, channel]))
        
        img[:, :, channel] = img[:, :, channel] - mymean
        img[:, :, channel] = img[:, :, channel] / mystdv

    return img

def load_classifiers():
    classifiers = []

    for channel in range(args.channels):

        # Load classifier from file
        classifier = joblib.load('classifiers/kmeans' + str(channel) + '.pkl')
        classifiers.append(classifier)

    return classifiers


def main(args):

    basedir = args.test_imgs_path
    num_histograms = args.num_test_pics
    channel = 0
    classifiers = load_classifiers()
    max_textons = args.max_textons
    n_clusters = args.num_textons
    weights = 1

    texton_hists = []

    for i in range(num_histograms):
        img_dir = basedir + "/" + str(i) + ".png"
        img = treXton.imread_opponent(img_dir)

        if args.local_standardize:
            for channel in range(args.channels):
                img = standardize(img, channel)

        hists_per_channel = []
        for channel in range(args.channels):
            classifier = classifiers[channel]        
                
            texton_hist = treXton.img_to_texton_histogram(img[:, :, channel], classifier, max_textons,
                                                      n_clusters, weights, args, channel)
            hists_per_channel.append(texton_hist)
            
        hists_per_channel = np.ravel(np.array(hists_per_channel)).astype(np.float32)
        color_histogram = False
        all_hists = hists_per_channel
        if color_histogram:
            # reorder data into for suitable for histogramming
            data = np.vstack((img[:, :, 0].flat, 
                              img[:, :, 1].flat,
                              img[:, :, 2].flat)).astype(np.uint8).T

            m = 4  # size of 3d histogram cube
            color_hist, edges = np.histogramdd(data, bins=m)
#            print np.ravel(color_hist / (640 * 480))
            #print hists_per_channel
            #print np.ravel(color_hist)
            all_hists = np.concatenate((hists_per_channel, np.ravel(color_hist)))

        texton_hists.append(all_hists)

    np.savetxt("mat_train_hists_cross.csv", texton_hists, delimiter=",", fmt='%d') 


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    global args
    args = parser.parse_args()
    main(args)
