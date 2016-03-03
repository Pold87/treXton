import numpy as np
import cv2
import configargparse
import treXton 
from treXtonConfig import parser

plot = 0
args = parser.parse_args()

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

img_path = args.test_imgs_path + str(0) + ".png"
old_frame = cv2.imread(img_path)
old_gray = treXton.imread_opponent_gray(img_path).astype('uint8')
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

picturenumbers = range(0, args.num_test_pics, 1)

posxy = np.array([280.0, 1070.0])

# Open file
fp = open("opticalflow.csv", "w")
fp_diff = open("opticalflow_diff.csv", "w")

for pic_num in picturenumbers:

    height = 1.45

    frame_path = args.test_imgs_path + str(pic_num) + ".png"
    frame = cv2.imread(frame_path)
    frame_gray = treXton.imread_opponent_gray(frame_path).astype('uint8')

    #frame = cv2.imread(frame_path)
    #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    diff = p0 - p1
    posdiffxy = np.mean(diff, 0)[0]
    posxy[0] += height * posdiffxy[0]
    posxy[1] += height * posdiffxy[1]

    fp_diff.write("%f,%f\n" % (height * posdiffxy[1], height * posdiffxy[0]))     
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]


        if plot:
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)

            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        fp.write("%f,%f\n" % (posxy[1], posxy[0])) 

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

fp.close()
cv2.destroyAllWindows()
