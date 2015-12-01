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

    rel = relocalize.Relocalizer(args.mymap)
    
    for i in range(args.num_pics):
        img = os.path.join(args.basedir, str(i) + ".png")
        
        if os.path.exists(img):

            coords = rel.calcLocationFromPath(img)
            coords[1] = y_width - coords[1]
            ids.append(i)
            xs.append(coords[0])
            ys.append(coords[1])
            print(i, coords)

    targets['id'] = ids
    targets['x'] = xs
    targets['y'] = ys

    targets.to_csv("boodschappen.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mymap", default="map.jpg", help="Path to the mat image")
    parser.add_argument("-b", "--basedir", default="imgs/", help="Path to the images")
    parser.add_argument("-s", "--num_pics", default=85, help="Amount of pictures", type=int)
    args = parser.parse_args()

    main(args)
