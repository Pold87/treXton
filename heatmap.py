from __future__ import division

import shlex
import subprocess
import cv2
import numpy as np

"""

This helper module creates heatmaps

"""

def create_centers(centers, matches):

    # TODO: Use HSV or something like that for creating the heatmap
    colors = ['red',
              'orange',
              'green',
              'purple',
              'DeepSkyBlue',
              'DeepSkyBlue4']



    strings = []


    print len(matches)
    heatmap_stepsize = 120 / (len(matches) - 1)

    print heatmap_stepsize

    
    for i, m in enumerate(matches):

        hsv_color = np.uint8([[[120 - (i * heatmap_stepsize), 255, 128]]])

        print hsv_color
        
        bgr_color = cv2.cvtColor(hsv_color,cv2.COLOR_HSV2BGR)
        b, g, r = bgr_color[0, 0]
        bgr_color = (int(b), int(g), int(r))
        
        x, y = centers[m]

#         color_str = str(x) + ',' + str(y) + ' ' + colors[i] # Using the predefined colormap
        color_str = str(x) + ',' + str(y) + \
                    ' rgb(' + str(b) + ',' + str(g) + ',' + str(r) + ')' # Using 'dynamic' heatmap colors
        strings.append(color_str)

    return ' '.join(strings)
        

def heatmap(img, centers, matches, filename='heatmap.png', method='Shepards'):
    
    height, width, c = img.shape

    center_strings_with_colors = create_centers(centers, matches)

    cl =  ['convert', '-size', str(width) + 'x' + str(height),
           'xc:', '-sparse-color', 
           method,
           center_strings_with_colors,
           filename]

    subprocess.call(cl)

    h_map = cv2.imread(filename, 0)
    return h_map
    


# heatmap_stepsize = 120 / (len(matches) - 1)


    #for i, m in enumerate(matches):

    #    hsv_color = np.uint8([[[120 - (i * heatmap_stepsize), 255, 128]]]);8
        
    #    bgr_color = cv2.cvtColor(hsv_color,cv2.COLOR_HSV2BGR)
    
    #    b, g, r = bgr_color[0, 0]

    #    bgr_color = (int(b), int(g), int(r))

    #    location_img = cv2.circle(location_img_c, centers[m], 90, bgr_color, -1)
