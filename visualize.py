import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import cv2

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

for i in range(start_pic, len(xs)):
    img_path = path + str(i) + ".jpg"
    pic = cv2.imread(img_path, 0)
    cv2.imshow("Bottom camera", pic)
    line.set_xdata(xs[max(0, i - 20):i])  # update the data
    line.set_ydata(ys[max(0, i - 20):i])
    cv2.waitKey(1)
    plt.pause(.3)
