import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import cv2

predictions = np.load("predictions.npy")[:100]
path = "/home/pold87/Downloads/imgs_first_flight/"

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 5000), ylim=(-2000, 2000))
line, = ax.plot([], [], lw=2)

plt.title('Predictions')

xs = predictions[:, 0]
ys = predictions[:, 1]

plt.ion()

for i in range(len(xs)):
    pic = cv2.imread(path + str(i) + ".jpg", 0)
    plt.imshow(pic)
    line.set_xdata(xs[max(0, i - 20):i])  # update the data
    line.set_ydata(ys[max(0, i - 20):i])
    cv2.waitKey(1)
    plt.pause(.1)
