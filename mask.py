import cv2
import numpy as np

img = cv2.imread("glass.png")
thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV)[1]

cv2.imwrite("glass_mask.png",thresh)


cv2.waitKey(0)
