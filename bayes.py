from __future__ import division, print_function

import numpy as np
import cv2


from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

w = 100.
h = 100.

p_pos_matrix = np.ones((w, h)) / (w * h)

def p_pos(x, y):
    return p_pos_matrix[x, y]


def p_measurement(p1, p2, p3, p4):
    return 100. * 100. * 100. * 100.


def measure(x, y):

    width_camera_field = 20
    height_camera_field = 20

    mcdonalds = Rectangle(0., 50., 50., 100.)
    camel = Rectangle(50., 50., 100., 100.)
    linux = Rectangle(0., 0., 50., 50.)
    ubuntu = Rectangle(50., 0., 100., 50.)

    cam = Rectangle(x - width_camera_field / 2,
                       y - height_camera_field / 2,
                       x + width_camera_field / 2,
                       y + height_camera_field / 2)

    m_area = area(mcdonalds, cam)
    c_area = area(camel, cam)
    l_area = area(linux, cam)
    u_area = area(ubuntu, cam)

    normalized_intersection = np.array([m_area, c_area, l_area, u_area]) / np.sum(np.array([m_area, c_area, l_area, u_area]))

    return normalized_intersection
    

# Likelihood function
def p_measurement_given_pos(p1, p2, p3, p4, x, y, tol=0.):

    normalized_intersection = measure(x, y)

    #print(normalized_intersection)


    if np.allclose(np.array([p1, p2, p3, p4]), normalized_intersection, atol=tol):
        return 1
    
    return 0


#p_measurement_given_pos(0.3, 0.2, 0.1, 0.4, 55, 80)    

def likelihood(p1, p2, p3, p4, x, y):
    return p_measurement_given_pos(p1, p2, p3, p4, x, y)
    
def p_pos_given_measurement(p1, p2, p3, p4, x, y):
    p = (p_measurement_given_pos(p1, p2, p3, p4, x, y) * p_pos(x, y)) / p_measurement(p1, p2, p3, p4)

    return p

#print(p_pos_given_measurement(0.0, 1.0, 0.0, 0.0, 80, 80))

def maximum_a_posteriori(p1, p2, p3, p4):

    x_max = 0
    y_max = 0

    #tolerances = [0., 0.01, 0.05, 0.1]
    tolerances = [0.1]
    for tol in tolerances:
        for x in range(int(w)):
            for y in range(int(h)):
                if p_measurement_given_pos(p1, p2, p3, p4, x, y, tol) == 1:
                    x_max = x
                    y_max = y
                    print("used tolerances was", tol)
                    return x_max, y_max
    return x_max, y_max
    
def main():
    measurement = (0.3, 0.2, 0.1, 0.4)
    #measurement = (0.0, 0.0, 0.0, 1.0)
    #measurement = (0.5, 0.5, 0.0, 0.0)
    print("measurement (from treXton classification) is", measurement)
    x, y = maximum_a_posteriori(*measurement)
    print("best position estimate (maximum a posteriori probability is)", "x:", x, "y:", y)
    print("this corresponds to the closest possible measurement of", measure(x, y))            

if __name__ == "__main__":
    main()


    
