from __future__ import division

import cv2
import numpy as np
from scipy import ndimage
import inspect, re
import matplotlib.pyplot as plt

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            print m.group(1), "is:", p

# Rotate between 0 and 90 degrees, based on (0, 0)
angle = 35

# original image
img = cv2.imread('../maze17.jpg', 0)
img_c = cv2.imread('../maze17.jpg', 1)

#rotation angle in degree
#img = ndimage.rotate(img, angle)
#img_c = ndimage.rotate(img_c, angle)

h, w = img.shape

corners = np.array([[[0], [0]],
                    [[h], [0]],
                    [[h], [w]], 
                    [[0], [w]]])


def make_rotation_matrix(angle):

    print angle

    M = np.matrix([[np.cos(angle), - np.sin(angle)],
                  [np.sin(angle),   np.cos(angle)]])

    print M

    return M


def rotate(angle, coordinates):

    M = make_rotation_matrix(np.radians(angle))
    new_coordinates = np.zeros_like(coordinates, dtype=np.float32)

    for i, c in enumerate(coordinates):
        
        rotated = M * c
        new_coordinates[i] = rotated

    return np.array(new_coordinates)

rotated = rotate(-20, corners).astype(np.int32)

print "Rotated", rotated

rotated_for_min = rotated.reshape(-1, 2)

print "Rotated", rotated

x_min, y_min = rotated_for_min.min(axis=0)

print "x_min", x_min
print "y_min", y_min

new_pic = np.zeros((h + np.fabs(y_min), w))
new_pic

def rotate_coordinates(xs, ys, theta):

    """
    This function rotates 2D coordinates by a specified angle.
    """
    
    xs_new = np.zeros(len(xs))
    ys_new = np.zeros(len(ys))
    
    for i, (x, y) in enumerate(zip(xs, ys)):

        new_x = np.cos(theta) * x - np.sin(theta) * y
        new_y = np.sin(theta) * x + np.cos(theta) * y

        xs_new[i] = new_x
        ys_new[i] = new_y

    return xs_new, ys_new

print rotate_coordinates([1606], [2415], 35)

min_window_width = 100
min_window_height = 100

pos_y = np.random.randint(0, h - min_window_height)
varname(pos_y)

dist_to_zero_x = np.fabs(1750 - pos_y)

varname(dist_to_zero_x)

begin_of_image_x = dist_to_zero_x /  np.tan(np.radians(angle))

varname(begin_of_image_x)


b = 1750 - begin_of_image_x
tmp = np.sin(np.radians(angle)) / b

varname(tmp)

c = tmp / np.cos(np.radians(angle))

varname(c)

width_of_image_x = c
varname(width_of_image_x)

pos_x = np.random.randint(begin_of_image_x, begin_of_image_x + 1)

pos_x = 500 

varname(pos_y)
varname(pos_x)

window_width = np.random.randint(min_window_width, w - pos_x)
window_height = np.random.randint(min_window_height, h - pos_y)

varname(window_width)
varname(window_height)

rect_pts = [(pos_y, pos_x), 
            (pos_y + window_height, pos_x),
            (pos_y + window_height, pos_x + window_width),
            (pos_y, pos_x + window_height)]

print rect_pts

region = img[pos_y : pos_y + window_height, pos_x : pos_x + window_width]
#rect = cv2.polylines(img, rect_pts, True, (255, 255, 255))

print rect_pts[0]
print rect_pts[3]

rect = cv2.rectangle(img_c, rect_pts[0], rect_pts[2], (122, 122, 122), 5)

#cv2.imshow('img1', img_c)
plt.imshow(img_c, 'Greys')
plt.show()

#cv2.waitKey()
#cv2.destroyAllWindows()
