import cv2
import numpy as np

img = cv2.imread("assets/Dog_Filter_assets/dog_filter_nose.png",0)
thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY)[1]

cv2.imwrite("assets/Dog_Filter_assets/dog_filter_nose_mask.png",thresh)
