import cv2
import numpy as np

cap = cv2.VideoCapture(1)
while True:
    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3))))
