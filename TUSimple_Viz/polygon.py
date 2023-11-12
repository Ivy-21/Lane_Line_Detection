# import required packages
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


# read each line of json file
json_gt = [json.loads(line) for line in open('data/label_data_0531.json')]

gt = json_gt[5]
#print("gt : ", gt)

gt_lanes = gt['lanes']
#print("gt_lanes : ", gt_lanes)

y_samples = gt['h_samples']
#print("y_samples : ", y_samples)

raw_file = gt['raw_file']
#print("raw_file : ", raw_file)


# see the image
img = cv2.imread('data/'+ raw_file)
#print("image shape : ", img.shape)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for lane in gt_lanes:
#     print("Lane : ", lane)
#     if x >= 0:
#         for (x, y) in zip(lane, y_samples):
#             print((x,y))


######## Draw Polygons ###########
pts = []

#[[pts.append([x, y]) for [x, y] in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]


for lane in gt_lanes:
    #print("lane : ", lane)
    #print("y_samples : ", y_samples)
    for [x, y] in zip(lane, y_samples):
        if x >= 0:
            print([x,y])
            pts.append([x,y])
    print("pts : ", pts)
    break
       
           


###########################################################
# print("type(pts) : ", type(pts))
# print(pts)

points = np.array(pts)
print("points.shape : ", points.shape)
points = points.reshape((-1, 1, 2))
print("points.shape ", points.shape)
# print(points)
#############################################################

# # # # # new points for polygon
# points_ = [[383, 313], [380, 382], [538, 528], [623, 545], [845, 475], [872, 401], [668, 352], [578, 333]]
# print("type(points_) : ", type(points_))
# print(points_)
# points = np.array(points_)
# print("points.shape : ", points.shape)
# points = points.reshape((-1, 1, 2))
# print("points.shape ", points.shape)
# # print(points)
###############################################################################################
# Attributes
isClosed = True
color = (255, 0, 0)
thickness = 2

# draw closed polyline

gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
img_vis = img.copy()

for lane in gt_lanes_vis:
    cv2.polylines(img_vis, np.int32([points]), isClosed, color, thickness)
    cv2.imshow("img_vis", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#################################################################################################

gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
img_vis = img.copy()

for lane in gt_lanes_vis:
    cv2.polylines(img_vis, np.int32([lane]), isClosed=True, color=(0,255,0), thickness=10)

cv2.imshow("img_vis", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()


# kernel = np.ones((5, 5), np.uint8)
# img_dilation = cv2.dilate(img_vis, kernel, iterations=1 )

# cv2.imshow('Dilation', img_dilation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite('dilatededimage.jpg', img_dilation)    

# #closing = cv2.morphologyEx(img_vis, cv2.MORPH_CLOSE, kernel)
# opening = cv2.morphologyEx(img_vis, cv2.MORPH_OPEN, kernel)
     
# # The mask and closing operation
# # is shown in the window 
# # cv2.imshow('Mask', img_vis)
# #cv2.imshow('Closing', closing)
# cv2.imshow('Opening', opening)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



mask = np.zeros_like(img)
colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255]]
for i in range(len(gt_lanes_vis)):
    cv2.polylines(mask, np.int32([gt_lanes_vis[i]]), isClosed=False,color=colors[i], thickness=10)


# create grey-scale label image
label = np.zeros((720,1280),dtype = np.uint8)
for i in range(len(colors)):
   label[np.where((mask == colors[i]).all(axis = 2))] = i+1

cv2.imshow("mask_img", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



# kernel = np.ones((7, 7), np.uint8)
# mask_img_dilation = cv2.dilate(mask, kernel, iterations=1 )

# cv2.imshow('Mask Dilation', mask_img_dilation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# masked = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow("Mask applied to Image", masked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
