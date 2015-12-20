#!/usr/bin/env python

import relocalize
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main(args):

    background_map = plt.imread(args.mymap)
    y_width, x_width, _ = background_map.shape
    
    targets = pd.DataFrame()

    ids = []
    xs = []
    ys = []
    num_matches = []

    rel = relocalize.Relocalizer(args.mymap)
    
    for i in range(args.num_pics):
        img = os.path.join(args.basedir, str(i) + ".png")
        
        if os.path.exists(img):

            coords = rel.calcLocationFromPath(img)
            coords[1] = y_width - coords[1]
            ids.append(i)
            xs.append(coords[0])
            ys.append(coords[1])
            num_matches.append(coords[2])
            print(i, coords)

    targets['id'] = ids
    targets['x'] = xs
    targets['y'] = ys
    targets['matches'] = num_matches

    targets.to_csv("sift_targets.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mymap", default="../draug/img/bestnewmat.png", help="Path to the mat image")
    parser.add_argument("-b", "--basedir", default="imgs/", help="Path to the images")
    parser.add_argument("-s", "--num_pics", default=2000, help="Amount of pictures", type=int)
    args = parser.parse_args()

    main(args)
