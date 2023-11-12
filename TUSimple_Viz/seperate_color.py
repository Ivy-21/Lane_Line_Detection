


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

# cv2.imshow("img_vis", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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

# cv2.imshow("mask_img", mask_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####################################################################################################################

gray = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)

# cv2.imshow("gray", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#####################################################################################################################

kernel = np.ones((9, 9), np.uint8)
dilation = cv2.dilate(gray, kernel, iterations=1 )

# cv2.imshow('Mask Dilation', dilation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

######################################################################################################################

# binarize the image
th = cv2.threshold(dilation,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


# cv2.imshow("th", th)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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

contours_array = np.array(contours)

contours_array = np.squeeze(contours_array)


###################################################################################################################

#https://stackoverflow.com/questions/55592105/how-to-blackout-the-background-of-an-image-after-doing-contour-detection-using-o

# Generate mask
mask = np.ones(gray.shape)
mask = cv2.drawContours(mask, contours, -1, 0, cv2.FILLED)

# Generate output
output = image.copy()
output[mask.astype(np.bool), :] = 0

# cv2.imwrite("output.png", output)

cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        

cv2.imshow('image hsv',hsv) 
cv2.waitKey(0)
cv2.destroyAllWindows()

##################################################################################
cnts = contours
coins = image.copy()

for cnt in cnts:
    #area = cv2.contourArea(cnt)
    if cv2.contourArea(cnt) > 800: # filter small contours
        x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
        cutted_contour = output[y:y+h,x:x+w]
        cv2.imshow('cutted contour',cutted_contour)
        # color = np.array(cv2.mean(cutted_contour)).astype(np.uint8)
        # print('Average color (BGR): ',color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 

        image_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        cut_image_hsv = output[y:y+h,x:x+w]

        cv2.imshow('cutted image hsv',image_hsv[y:y+h,x:x+w]) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#########################################################################################################

        color = np.array(cv2.mean(cut_image_hsv)).astype(np.uint8)
        color_map = np.array(color[0:-1])
        print('Average color (BGR): ', color_map)

        # moment = cv2.moments(cnt)
        # print("moments : ", moment)

        if ((color_map[0] >= 10) and (color_map[1] >= 10) and (color_map[2 ] >= 10)):
            print(color_map[0])
            print(color_map[1])
            print(color_map[2])
            print("white")
            cv2.drawContours(coins , [cnt], -1, (255, 255, 255), -1)
        else:
            print(color_map[0])
            print(color_map[1])
            print(color_map[2])
            print("yellow")
            cv2.drawContours(coins , [cnt], -1, (0, 255, 255), -1)

cv2.imshow('coins', coins)
cv2.waitKey(0)
cv2.destroyAllWindows()

        
#########################################################################################################


