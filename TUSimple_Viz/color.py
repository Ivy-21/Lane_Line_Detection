


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

print("image np array : ", np.array(image))

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

###################################################################################

# https://stackoverflow.com/questions/54316588/get-the-average-color-inside-a-contour-with-open-cv

cnts = contours
for cnt in cnts:
    if cv2.contourArea(cnt) >800: # filter small contours
        x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('cutted contour',img[y:y+h,x:x+w])
        color = np.array(cv2.mean(img[y:y+h,x:x+w])).astype(np.uint8)
        print('Average color (BGR): ',color)
        cv2.waitKey(0)

###################################################################################

# https://stackoverflow.com/questions/51614092/how-to-find-colour-of-contours-in-opencv-with-python


# # if you want cv2.contourArea >1, you can just comment line bellow
# cnts = np.array(contours)[[cv2.contourArea(c)>10 for c in contours]]
# grains = [np.int0(cv2.boxPoints(cv2.minAreaRect(c))) for c in cnts]
# centroids =[(grain[2][1]-(grain[2][1]-grain[0][1])//2, grain[2][0]-(grain[2][0]-grain[0][0])//2) for grain in grains]

# colors = [centroid for centroid in centroids]
# print("colors : ", colors)

####################################################################

from shapely.geometry import Polygon
from shapely import wkt

coco_json = []


for i, contour in enumerate(contours_array):
    #print("contour : ", contour)
    #print("Shape of contour : ", contour.shape)
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

####################



# def grab_rgb(image, c):
#     pixels = []

#     # TODO: Convert to real code
#     # Detect pixel values (RGB)
#     mask = np.zeros_like(image)
#     cv2.drawContours(mask, c, -1, color=255, thickness=-1)

#     points = np.where(mask == 255)

#     for point in points:
#         pixel = (image[point[1], point[0]])
#         pixel = pixel.tolist()
#         pixels.append(pixel)

#     pixels = [tuple(l) for l in pixels]
#     car_color = (pixels[1])

#     r = car_color[0]
#     g = car_color[1]
#     b = car_color[2]

#     pixel_string = '{0},{1},{2}'.format(r, g, b)

#     return pixel_string


# # look for motion
# motion_found = False
# biggest_area = 0

#     # examine the contours, looking for the largest one
# for c in contours:
#     (x, y, w, h) = cv2.boundingRect(c)
#     # get an approximate area of the contour
#     found_area = w * h
#     # find the largest bounding rectangle
#     if (found_area > MIN_AREA) and (found_area > biggest_area):
#         biggest_area = found_area
#         motion_found = True

#         if not is_nighttime():
#             rgb = grab_rgb(image, c)
#         else:
#             rgb = 'nighttime'


####################

color_list = [[77, 89, 76], [69, 81, 71], [64, 91, 73], [45, 74, 45]]

def color2label(color_map):
    a = 0   
    for j,w in enumerate(color_map):  
        print(w[j])
    #label = np.ones(color_map.shape[:2])  #self.ignore_label
        for i, v in enumerate(color_list):
            # print(v[0])
            # print(color_map[0][i][0])
            if color_map[0][j][0] == 77 and color_map[0][j][1] == 89 and color_map[0][j][2] == 76:
                a+=1
    print(a)
        #break
            # print(color_map[0][0] == v)
            # break
            # if v == 2 and i <= 12: # barrier
            #     label[(color_map == v).sum(2)==3] = 0
            # elif i >= 13 and i <= 25: # road
            #     label[(color_map == v).sum(2)==3] = 1
            # elif i >= 30 and i <= 34: # people
            #     label[(color_map == v).sum(2)==3] = 2
            # else:
            #     label[(color_map == v).sum(2)==3] = 3

        # return label.astype(np.uint8)
print(img.shape)
print(len(np.array(img )))
# color2label(np.array(image))