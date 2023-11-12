# import required packages
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


# read each line of json file
json_gt = [json.loads(line) for line in open('data/label_data_0531.json')]
gt = json_gt[0]
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
    cv2.polylines(img, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=6)
    

cv2.imshow("img_vis", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################################################################################################

mask_ = np.zeros_like(img)
colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255]]

label = np.zeros((720,1280),dtype = np.uint8)

kernel = np.ones((9, 9), np.uint8)

t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

points = []



for lane in gt_lanes_vis:
    for i in range(len(gt_lanes_vis)):
        cv2.polylines(mask_, np.int32([gt_lanes_vis[i]]), isClosed=False,color=colors[i], thickness=3)
       
    
    
    for i in range(len(colors)):
        label[np.where((mask_ == colors[i]).all(axis = 2))] = i+1
        gray = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
       


    dilation = cv2.dilate(gray, kernel, iterations=1 )
    th = cv2.threshold(dilation,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    edge = cv2.Canny(th, t_lower, t_upper)


    indices = np.where(edge != [0])
    coordinates = list(zip(indices[0], indices[1]))


    points.append(coordinates)
    points_array = np.array(points)
    points_array = np.squeeze(points_array)
    print(points_array.shape)
    print(points_array)


  

    



cv2.imshow("mask_img", mask_)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Mask Dilation', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("th", th)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()



  
