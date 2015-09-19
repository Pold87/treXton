import cv2
import numpy as np
import texton_helpers

from skimage.util import view_as_windows

img_path = "maze17.jpg"
img_path = "guy.jpg"
img_path = "/home/pold87/Desktop/Bilder/P1080334.JPG"
cv_img = cv2.imread(img_path, 0)

print cv_img

training_image = cv2.imread(img_path, 0)

num_patches_h = 2
num_patches_w = 3


h, w = training_image.shape

patches = texton_helpers.sliding_window(training_image,
                                        (h / num_patches_h, w / num_patches_w), 
                                        None, 
                                        True)

blurred = cv2.blur((patches[0], 20))
cv2.imwrite('tlpic.jpg', blurred)

