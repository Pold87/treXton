import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import scipy
import cv2
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
    AnnotationBbox
from matplotlib._png import read_png

predictions = np.load("predictions.npy")
path = "/home/pold/Documents/imgs_first_flight/"

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 5000), ylim=(-2000, 2000))
line, = ax.plot([], [], lw=2)

plt.title('Predictions')

xs = predictions[:, 0]
ys = predictions[:, 1]

plt.ion()

start_pic = 50

arr_lena = read_png("img/minidrone.png")
imagebox = OffsetImage(arr_lena, zoom=1)
background_map = plt.imread("../draug/img/map_3_cropped.png")
plt.imshow(background_map, zorder=0, extent=[0, 5000, -2000, 2000])

for i in range(start_pic, len(xs)):
    img_path = path + str(i) + ".jpg"

    xy = (xs[i], ys[i])
    
    ab = AnnotationBbox(imagebox, xy,
    xycoords='data',
    pad=0.0,
    frameon=False)
    
    pic = cv2.imread(img_path, 0)
    cv2.imshow("Bottom camera", pic)
    line.set_xdata(xs[max(start_pic, i - 13):i])  # update the data
    line.set_ydata(ys[max(start_pic, i - 13):i])
    
    cv2.waitKey(1)
    plt.pause(.3)
    if i != start_pic:
        drone_artist.remove()
        
    drone_artist = ax.add_artist(ab)

