# import required packages
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import imutils

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


# read each line of json file
json_gt = [json.loads(line) for line in open('data/label_data_0531.json')]
gt = json_gt[9]
gt_lanes = gt['lanes']
y_samples = gt['h_samples']
raw_file = gt['raw_file']


# see the image
image = cv2.imread('data/'+ raw_file)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################################################################################################

gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
img = image.copy()

for lane in gt_lanes_vis:
    print("lane : ", len(lane))
    print("type of lane : ", type(lane))
    cv2.polylines(img, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=6)

cv2.imshow("img_vis", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#####################################################################################################################

mask_ = np.zeros_like(img)
colors = [[0,255,0],[0,255,0],[0,255,0],[0,255,0]]
for i in range(len(gt_lanes_vis)):
    cv2.polylines(mask_, np.int32([gt_lanes_vis[i]]), isClosed=False,color=colors[i], thickness=3)


# create grey-scale label image
label = np.zeros((720,1280),dtype = np.uint8)
for i in range(len(colors)):
   label[np.where((mask_ == colors[i]).all(axis = 2))] = i+1

cv2.imshow("mask_img", mask_)
cv2.waitKey(0)
cv2.destroyAllWindows()

####################################################################################################################

# img = cv2.imread('hand_drawn_contours.jpg',1)
gray = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################################################################################################

kernel = np.ones((9, 9), np.uint8)
dilation = cv2.dilate(gray, kernel, iterations=1 )

cv2.imshow('Mask Dilation', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

######################################################################################################################

# binarize the image
th = cv2.threshold(dilation,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

cv2.imshow("th", th)
cv2.waitKey(0)
cv2.destroyAllWindows()

#######################################################################################################################


# Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold
  
# Applying the Canny Edge filter
edge = cv2.Canny(th, t_lower, t_upper)
  
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
 #########################################################################################################################



indices = np.where(edge != [0])

coordinates = list(zip(indices[0], indices[1]))
# print("coordinates : ",coordinates)
# print(type(coordinates))

################################################################################################################
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
dilate = cv2.dilate(edge, kernel_ellipse, iterations=1)

contours, hierarchy  = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#contours = contours[0] 

#contours_ = contours[1] if imutils.is_cv3() else contours[0]

# print("contours : ",contours)
# print("type of contours : ",type(contours))

contours_array = np.array(contours)
# print("type of contours_array : ",type(contours_array))

contours_array = np.squeeze(contours_array)
# print(contours_array.shape)


coins = img.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area > 1000:
         cv2.drawContours(coins , [c], -1, (0, 0, 255), -1)

cv2.imshow('coins', coins)
cv2.waitKey(0)
cv2.destroyAllWindows()

####################################################################

from shapely.geometry import Polygon
from shapely import wkt

coco_json = []


for i, contour in enumerate(contours_array):
    print("contour : ", contour)
    print("Shape of contour : ", contour.shape)
    x,y,w,h = cv2.boundingRect(contour)
    img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    img = cv2.drawContours(img,[contour],0,(0,255,255),-1)
    coco_json.append({"bbox" : [x,y,w,h], "category_id": 1, "id": i})
    #coco_json.append({"bbox" : [x,y,w,h], "category_id": 1, "id": i, "points" : [contour]})
    #polygon = Polygon(contour)
    #polygon = wkt.loads(contour)
    #break

with open("coco_format.json", "w") as outfile:
    json.dump(coco_json, outfile)

# cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
cv2.imshow("result",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # finding contours
# contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # create a blank mask
# bw_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
# # draw selected ROI to form the mask
# bw_mask = cv2.drawContours(bw_mask,contours,1,255, -1)

# cv2.imshow("Threshold_mask", bw_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ####################################################################################################################

# # dilate both the image and the mask
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
# dilate_th = cv2.dilate(th, kernel, iterations=1)
# #dilate_mask = cv2.dilate(bw_mask, kernel, iterations=1)

# # using the dilated mask, retain the dilated ROI
# #im1 = cv2.bitwise_and(dilate_th, dilate_th, mask = dilate_mask)



# #dilate_mask_inv = cv2.bitwise_not(dilate_mask)
# #im2 = cv2.bitwise_and(th, th, mask = dilate_mask_inv)
# #res = cv2.add(im1, im2)


# # cv2.imshow("Final", res)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()