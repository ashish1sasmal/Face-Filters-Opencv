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

def width(x,y):
    w =dist.euclidean(x,y)
    return int(w)

def calc_angle(dx,dy):
    angle = np.degrees(np.arctan2(dy,dx))-180
    return angle

def shift(w,h,angle,lx,ly):
    x = lx
    y = ly-90
    return (x,y)


l_ear = cv2.imread("assets/Dog_Filter_assets/dog_filter_left_ear.png")
r_ear = cv2.imread("assets/Dog_Filter_assets/dog_filter_right_ear.png")
nossy = cv2.imread("assets/Dog_Filter_assets/dog_filter_nose.png")

l_ear_mask = cv2.imread("assets/Dog_Filter_assets/dog_filter_left_ear_mask.png")
r_ear_mask = cv2.imread("assets/Dog_Filter_assets/dog_filter_right_ear_mask.png")
nossy_mask = cv2.imread("assets/Dog_Filter_assets/dog_filter_nose_mask.png")

img = cv2.imread("Tests/"+sys.argv[1])
img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

rects = detect(gray,1)


shape = None
x=0
y=0
w=0
h=0
(le1, le2)  = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(re1, re2) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(n1, n2) = (31,36)

print(face_utils.FACIAL_LANDMARKS_IDXS)

output = img.copy()
for (i,rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    (x,y,w,h) = face_utils.rect_to_bb(rect)

    left = shape[le1:le2]
    right = shape[re1:re2]
    nose = shape[n1:n2]

    dy =   right[3][1]-right[1][1]
    dx = right[3][0]-right[1][0]
    re_angle = calc_angle(dx,dy)
    # print(dx,dy,re_angle)

    dy =   left[1][1]-left[3][1]
    dx = left[1][0]-left[3][0]
    le_angle = calc_angle(dx,dy)

    dy =   nose[0][1]-nose[4][1]
    dx = nose[0][0]-nose[4][0]
    nose_angle = calc_angle(dx,dy)
    print(nose_angle)

    wid = width(shape[17],shape[26])
    ear_width = wid//2
    nose_width =int(width(nose[0],nose[4])*2)

    # print(le_angle)
    # print(angle)

    left_ear = imutils.rotate_bound(l_ear,le_angle)
    right_ear = imutils.rotate_bound(r_ear,180+re_angle)
    Nose = imutils.rotate_bound(nossy,nose_angle)

    left_ear = imutils.resize(left_ear,width=ear_width)
    right_ear = imutils.resize(right_ear,width=ear_width)
    Nose = imutils.resize(Nose,width=nose_width)

    # ear_height = left_ear.shape[0]

    lm  = cv2.cvtColor(l_ear_mask,cv2.COLOR_BGR2GRAY)
    lm = cv2.threshold(lm.copy(),0,255,cv2.THRESH_BINARY)[1]
    lm = imutils.rotate_bound(lm,le_angle)
    lm = imutils.resize(lm,width = ear_width, inter=cv2.INTER_NEAREST)

    rm  = cv2.cvtColor(r_ear_mask,cv2.COLOR_BGR2GRAY)
    rm = cv2.threshold(rm.copy(),0,255,cv2.THRESH_BINARY)[1]
    rm = imutils.rotate_bound(rm,re_angle)
    rm = imutils.resize(rm,width = ear_width, inter=cv2.INTER_NEAREST)

    nm  = cv2.cvtColor(nossy_mask,cv2.COLOR_BGR2GRAY)
    nm = cv2.threshold(nm.copy(),0,255,cv2.THRESH_BINARY)[1]
    nm = imutils.rotate_bound(nm,nose_angle)
    nm = imutils.resize(nm,width = nose_width, inter=cv2.INTER_NEAREST)

    nh = width(shape[30],shape[33])

    output = overlay_image(output, left_ear, lm,(int(x+0.7*w),int(y-h/2.2)))
    output = overlay_image(output, right_ear, rm,(x,int(y-h/2.2)))
    output = overlay_image(output, Nose, nm,(int(nose[0][0]-nose_width*0.2),int(shape[30][1]-nh*1.3)))

cv2.imwrite(f"Results/result{sys.argv[2]}.jpg",output)
print("Boom! Boom! Done!")
cv2.imshow("image",output)
cv2.waitKey(0)
