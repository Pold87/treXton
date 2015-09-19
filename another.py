import cv2
import matplotlib.pyplot as plt

filename = '0.png'

h_map = cv2.imread(filename, 3)
h_map = plt.imread(filename, 1)


plt.imshow(h_map)
plt.show()
