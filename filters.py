import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from overlay import overlay_image
import sys
from scipy.spatial import distance as dist
import math


cv2.namedWindow("image",cv2.WINDOW_NORMAL)

def glass_width(le,re):
    w =dist.euclidean(re,le)
    return int(w)

def shift(w,h,angle,lx,ly):
    d = math.pi/180
    angle*=-1
    x = int(lx-int(math.cos(d*angle)*w*0.255))
    y = int(ly-int(math.cos(d*angle)*h*0.6))
    return (x,y)


glass = cv2.imread("assets/glass.png")
mask = cv2.imread("assets/glass_mask.png")
img = cv2.imread("Tests/"+sys.argv[1])
img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

rects = detect(gray,2)


shape = None
x=0
y=0
w=0
h=0
(le1, le2)  = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(re1, re2) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
output = img.copy()
for (i,rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    (x,y,w,h) = face_utils.rect_to_bb(rect)

    left = shape[le1:le2]
    right = shape[re1:re2]

    le_center = left.mean(axis=0).astype("int")
    re_center = right.mean(axis=0).astype("int")

    wid = glass_width(le_center,re_center)

    g_width = int(2*wid)

    # compute the angle between the eye

    dy = re_center[1]-le_center[1]
    dx = re_center[0]-le_center[0]
    angle = np.degrees(np.arctan2(dy,dx))-180

    # print(angle)

    glass1 = imutils.rotate_bound(glass,angle)

    glass1 = imutils.resize(glass1,width=g_width)

    glass_height = glass1.shape[0]

    mask1  = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask1 = cv2.threshold(mask1.copy(),0,255,cv2.THRESH_BINARY)[1]
    mask1 = imutils.rotate_bound(mask1,angle)

    mask1 = imutils.resize(mask1,width = g_width, inter=cv2.INTER_NEAREST)

    output = overlay_image(output, glass1, mask1,shift(g_width,glass_height,angle,re_center[0],re_center[1]))


cv2.imwrite(f"Results/result{sys.argv[2]}.jpg",output)
print("Boom! Boom! Done!")
cv2.imshow("image",output)
cv2.waitKey(0)
