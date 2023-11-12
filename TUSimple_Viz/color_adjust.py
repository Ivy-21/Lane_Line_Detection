


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

#print("image np array : ", np.array(image))

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

##################################################################################
cnts = contours
coins = img.copy()

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


        ####################################

        # cut_output = output.copy()

        # cx = x+w//2
        # cy = y+h//2
        # cr  = max(w,h)//2

        # dr = 10
        # for i in range(0,10):
        #     r = cr+i*dr
        #     cv2.rectangle(cut_output, (cx-r, cy-r), (cx+r, cy+r), (0,255,0), 1)
        #     croped = cut_output[cy-r:cy+r, cx-r:cx+r]
        #     cv2.imshow("croped{}".format(i), croped)


        #cutted_contour_image = output[y:y+h,x:x+w]

       


        image_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        #image_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HLS)
        #cut_image_hsv = image_hsv[y:y+h,x:x+w]
        cut_image_hsv = output[y:y+h,x:x+w]

        cv2.imshow('cutted image hsv',image_hsv[y:y+h,x:x+w]) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #print("image hsv array : ", np.array(image_hsv))

#########################################################################################################

        color = np.array(cv2.mean(cutted_contour)).astype(np.uint8)
        color_map = np.array(color[0:-1])
        print('Average color (BGR): ', color_map)

        if ((color_map[0] >= 10) and (color_map[1] >= 10) and (color_map[2 ] >= 10)):
            print(color_map[0])
            print(color_map[1])
            print(color_map[2])
            print("yellow")
            cv2.drawContours(coins , [cnt], -1, (255, 255, 0), -1)
        else:
            print(color_map[0])
            print(color_map[1])
            print(color_map[2])
            print("white")
            cv2.drawContours(coins , [cnt], -1, (0, 0, 255), -1)

cv2.imshow('coins', coins)
cv2.waitKey(0)
cv2.destroyAllWindows()

        

        

#########################################################################################################


#         contour_color = None
#         yellow_pixels = []
#         white_pixels = []
#         color_map = np.array(cut_image_hsv) 
#         #color_map = np.array(cutted_contour)
#         #print("color map : ", color_map)

#         for j,w in enumerate(color_map):

#             # print("coloe_map : ", color_map)
            

#             # print(color_map[j][0][0])
#             # print(color_map[j][0][1])
#             # print(color_map[j][0][2])

#             # if (color_map[j][0][0] == 0 and color_map[j][0][1] == 255 and color_map[j][0][1] == 0):
#             #     contour_color = "white"
#             #     cv2.drawContours(coins , [cnt], -1, (0, 255, 0), -1)
#             # else:
#             #     contour_color = "yellow"
#             #     cv2.drawContours(coins , [cnt], -1, (0, 0, 255), -1)
            
           
#             if (color_map[j][0][0] >= 25 and color_map[j][0][0] <= 35) and (color_map[j][0][1] >= 50 and color_map[j][0][1] <= 255) and (color_map[j][0][2] >= 70 and color_map[j][0][2] <= 255) :
#                 print(color_map[j][0][0] >= 25 and color_map[j][0][0] <= 35) and (color_map[j][0][1] >= 50 and color_map[j][0][1] <= 255) and (color_map[j][0][2] >= 70 and color_map[j][0][2] <= 255)
#                 contour_color = "yellow"
#                 yellow_pixels.append(1)
#                 #cv2.drawContours(coins , [cnt], -1, (0, 255, 255), -1)
            
#             elif (color_map[j][0][0] >= 0 and color_map[j][0][0] <= 180) and (color_map[j][0][1] >= 0 and color_map[j][0][1] <= 18) and (color_map[j][0][2] >= 231 and color_map[j][0][2] <= 255) :
#                 contour_color = "white"
#                 white_pixels.append[1]
#                 #cv2.drawContours(coins , [cnt], -1, (0, 0, 255), -1)
           
#             else: 
#                 contour_color = "other"
#                 cv2.drawContours(coins , [cnt], -1, (0, 255, 0), -1)
#         #break

#         # print("yellow : ", len(yellow_pixels))
#         # print('white : ', len(white_pixels))
        


#         #cv2.drawContours(coins , [cnt], -1, (0, 0, 255), -1)
#     #break

# cv2.imshow('coins', coins)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # coins = img.copy()
# # for c in contours:
# #     area = cv2.contourArea(c)
# #     if area > 1000:
# #          cv2.drawContours(coins , [c], -1, (0, 0, 255), -1)

# # cv2.imshow('coins', coins)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

    
  
    
    

       
            #print(w[j])
            #break
    
    #     for i, v in enumerate(color_list):
    #         # print(v[0])
    #         # print(color_map[0][i][0])
    #         if color_map[0][j][0] == 77 and color_map[0][j][1] == 89 and color_map[0][j][2] == 76:
    #             a+=1
    # print(a)



   


####################################################################################

# cnts = contours
# for cnt in cnts:
#     if cv2.contourArea(cnt) >800: # filter small contours
#         x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
#         cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.imshow('cutted contour',output[y:y+h,x:x+w])

#         #print("cont_img : ", np.array(output[y:y+h,x:x+w]))
#         color_ = np.array(cv2.mean(output[y:y+h,x:x+w])).astype(np.uint8)
#         color = color_[0:-1]
#         #print('Average color (BGR): ',color)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         out = output[y:y+h,x:x+w]
#         original = out.copy()
#         image_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)

#         lower = np.array([25, 50, 70], dtype="uint8")
#         upper = np.array([35, 255, 255], dtype="uint8")

#         mask = cv2.inRange(image_hsv, lower, upper)

#         cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = cnts[0] if len(cnts) == 2 else cnts[1]

     
#         for c in cnts:
#             x,y,w,h = cv2.boundingRect(c)
#             cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
#             cv2.imshow('cutted contour',original[y:y+h,x:x+w])
#             color_f = np.array(cv2.mean(original[y:y+h,x:x+w])).astype(np.uint8)
#             final_color = color_f[0:-1]
#             #print('Average color (BGR): ',final_color)
#         # cv2.imshow('mask', mask)
#         # cv2.imshow('original', original)
#         # cv2.waitKey()


#-------------------------------------------------------------------------------------------------------
      
 

# cont_img = img.copy()
# for c in contours:
#     area = cv2.contourArea(c)
#     #print("area : ", area)
#     if area > 1000:
#          cv2.drawContours(cont_img , [c], -1, (0, 0, 255), -1)


        
# cv2.imshow('coins', coins)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
##################################################################


# cnts = contours
# for cnt in cnts:
#     if cv2.contourArea(cnt) >800: # filter small contours
#         x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.imshow('cutted contour',img[y:y+h,x:x+w])

#         print("cont_img : ", np.array(img[y:y+h,x:x+w]))
#         color_ = np.array(cv2.mean(img[y:y+h,x:x+w])).astype(np.uint8)
#         color = color_[0:-1]
#         print('Average color (BGR): ',color)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

######################################################____________________________________________________________
# cnts = contours
# for cnt in cnts:
#     if cv2.contourArea(cnt) >800: # filter small contours
#         x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
#         cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.imshow('cutted contour',output[y:y+h,x:x+w])
#         color = np.array(cv2.mean(output[y:y+h,x:x+w])).astype(np.uint8)
#         print('Average color (BGR): ',color)
        
#         print("cont_img : ", np.array(output[y:y+h,x:x+w]))
#         cv2.waitKey(0)


#         output = output[y:y+h,x:x+w]
#         original = output.copy()
#         image_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

#         lower = np.array([25, 50, 70], dtype="uint8")
#         upper = np.array([35, 255, 255], dtype="uint8")
        
#         mask = cv2.inRange(image_hsv, lower, upper)

#         cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = cnts[0] if len(cnts) == 2 else cnts[1]

     
#         for c in cnts:
#             x,y,w,h = cv2.boundingRect(c)
#             cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
#             cv2.imshow('cutted contour',output[y:y+h,x:x+w])
#             color_f = np.array(cv2.mean(output[y:y+h,x:x+w])).astype(np.uint8)
#             final_color = color_f[0:-1]
#             #print('Average color (BGR): ',final_color)

#         # cv2.drawContours(original , [c], -1, (0, 0, 255), -1)
    
#         cv2.imshow('mask', mask)
#         cv2.imshow('original', original)
#         cv2.waitKey()
        
######################################################____________________________________________________________
# upper_white = np.array([180, 18, 255], dtype="uint8")
# lower_white = np.array([0, 0, 231], dtype="uint8")

# def color2label(color_map):




####################

# color_list = [[77, 89, 76], [69, 81, 71], [64, 91, 73], [45, 74, 45]]

# def color2label(color_map):
#     a = 0   
#     for j,w in enumerate(color_map):  
#         print(w[j])
#     #label = np.ones(color_map.shape[:2])  #self.ignore_label
#         for i, v in enumerate(color_list):
#             # print(v[0])
#             # print(color_map[0][i][0])
#             if color_map[0][j][0] == 77 and color_map[0][j][1] == 89 and color_map[0][j][2] == 76:
#                 a+=1
#     print(a)
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
# print(img.shape)
# print(len(np.array(img )))
# color2label(np.array(image))