


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

######################################################################

output_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
#output_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HLS)

cv2.imshow('output_hsv', output_hsv )
cv2.waitKey(0)
cv2.destroyAllWindows()

##################################################################################

for cnt in contours:

    #area = cv2.contourArea(cnt)
    if cv2.contourArea(cnt) >800: # filter small contours
        x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
        cv2.rectangle(output_hsv,(x,y),(x+w,y+h),(0,255,0),2)

        # center_x = int(w/2)
        # center_y = int(h/2)

        # (b, g, r) = output_hsv[np.int16(center_y), np.int16(center_x)]
        # print("b : ", b)
        # print("g : ", g)
        # print("r : ", r)


        cutted_hsv = output_hsv[y:y+h,x:x+w]
        cv2.imshow('cutted hsv',output_hsv[y:y+h,x:x+w])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # color_map = np.array(cutted_hsv)
        # print("color_map : ", color_map)

        image_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        cut_image_hsv = image_hsv[y:y+h,x:x+w]
        #cut_image_hsv = output[y:y+h,x:x+w]

        cv2.imshow('cut image hsv',image_hsv[y:y+h,x:x+w])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # M = cv2.moments(cnt)
        # if M['m00'] != 0:
        #     cx = int(M['m10']/M['m00'])
        #     cy = int(M['m01']/M['m00'])
        # print(f"x: {cx} y: {cy}")
        # cv2.drawContours(image, [i], -1, (0, 255, 0), 2)

        # print(output_hsv[cx][cy])


    #     for i in contours:
    # M = cv.moments(i)
    # if M['m00'] != 0:
    #     cx = int(M['m10']/M['m00'])
    #     cy = int(M['m01']/M['m00'])
    #     cv.drawContours(image, [i], -1, (0, 255, 0), 2)
    #     cv.circle(image, (cx, cy), 7, (0, 0, 255), -1)
    #     cv.putText(image, "center", (cx - 20, cy - 20),
    #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # print(f"x: {cx} y: {cy}")

        #print("image hsv array : ", np.array(image_hsv))

        contour_color = None
        # # yellow_pixels = []
        # # white_pixels = []
        # color_map = np.array(cut_image_hsv) 
        # print("color_map[center_x][center_y]:", color_map[center_x][center_y])
        # #print("color map : ", color_map.shape)

        # for j,w in enumerate(color_map):
        #     if (color_map[j][0][0] >= 25 and color_map[j][0][0] <= 35) and (color_map[j][0][1] >= 50 and color_map[j][0][1] <= 255) and (color_map[j][0][2] >= 70 and color_map[j][0][2] <= 255) :
        #         contour_color = "yellow"
        #         #yellow_pixels.append(1)
        #         #cv2.drawContours(coins , [cnt], -1, (0, 255, 255), -1)
            
        #     elif (color_map[j][0][0] >= 0 and color_map[j][0][0] <= 180) and (color_map[j][0][1] >= 0 and color_map[j][0][1] <= 18) and (color_map[j][0][2] >= 231 and color_map[j][0][2] <= 255) :
        #         contour_color = "white"
        #         #white_pixels.append[1]
        #         #cv2.drawContours(coins , [cnt], -1, (0, 0, 255), -1)
           
        #     else: 
        #         contour_color = "other"
        #         #cv2.drawContours(coins , [cnt], -1, (0, 255, 0), -1)

#         print("yellow : ", len(yellow_pixels))
#         print('white : ', len(white_pixels))
        


      


################################################################################

# # yellow color 
# lower_yellow = np.array([25,100,100])
# upper_yellow = np.array([30,255,255])

# # white color 
# lower_white = np.array([0,0,231])
# upper_white = np.array([180,18,255])


# yellow = cv2.inRange(output_hsv, lower_yellow, upper_yellow)
# white = cv2.inRange(output_hsv, lower_white, upper_white)


# cnts_yellow = cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnts_yellow = imutils.grab_contours(cnts_yellow)

# cnts_white = cv2.findContours(white,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnts_white = imutils.grab_contours(cnts_white)

# #cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



# coins = img.copy()
# for c in cnts_white:
#     area = cv2.contourArea(c)
#     if area > 1000:
#          cv2.drawContours(coins , [c], -1, (0, 0, 255), -1)

# cv2.imshow('coins', coins)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
###############################################################################################

# for cnt in contours:
#     #area = cv2.contourArea(cnt)
#     if cv2.contourArea(cnt) >800: # filter small contours
#         x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
#         cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
#         cutted_contour = output[y:y+h,x:x+w]
#         cv2.imshow('cutted contour',output[y:y+h,x:x+w])
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()



##################################################################################



 

##################################################################################
# cnts = contours
# coins = img.copy()

# for cnt in cnts:
#     #area = cv2.contourArea(cnt)
#     if cv2.contourArea(cnt) >800: # filter small contours
#         x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
#         cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.imshow('cutted contour',output[y:y+h,x:x+w])
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         image_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
#         cut_image_hsv = image_hsv[y:y+h,x:x+w]
#         #cut_image_hsv = output[y:y+h,x:x+w]

#         cv2.imshow('cut image hsv',image_hsv[y:y+h,x:x+w])
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()



        # cutted_contour_image = output[y:y+h,x:x+w]
        # cutted_contour_image_copy = cutted_contour_image.copy()

        # cutted_image_hsv = cv2.cvtColor(cutted_contour_image, cv2.COLOR_BGR2HSV)
        # cutted_gray = cv2.cvtColor(cutted_image_hsv, cv2.COLOR_BGR2GRAY)
        # cutted_th = cv2.threshold(cutted_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # cutted_contours, hierarchy  = cv2.findContours(cutted_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for c in cutted_contours:
        #     if cv2.contourArea(cnt) >800: # filter small contours
        #         x_,y_,w_,h_ = cv2.boundingRect(c) # offsets - with this you get 'mask'
        #         cv2.rectangle(cutted_contour_image_copy,(x,y),(x+w,y+h),(0,255,0),2)
        #         cv2.imshow('cutted_contour_image_copy',cutted_contour_image_copy[y_:y_+h_,x_:x_+w_])
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()


        # cv2.imshow('cutted_th',cutted_th)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


#########################################################################################################


#         lower = np.array([25, 50, 70], dtype="uint8")
#         upper = np.array([35, 255, 255], dtype="uint8")

#         mask = cv2.inRange(image_hsv, lower, upper)

#         cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#         original = cut_image_hsv.copy()

     
#         for c in cnts:
#             x,y,w,h = cv2.boundingRect(c)
#             cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
#             cv2.imshow('cutted contour',original[y:y+h,x:x+w])
#             color_f = np.array(cv2.mean(original[y:y+h,x:x+w])).astype(np.uint8)
#             final_color = color_f[0:-1]
#             #print('Average color (BGR): ',final_color)
#         cv2.imshow('mask', mask)

       


        # image_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        # cut_image_hsv = image_hsv[y:y+h,x:x+w]
        # #cut_image_hsv = output[y:y+h,x:x+w]

        # cv2.imshow('cut image hsv',image_hsv[y:y+h,x:x+w])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # print("image hsv array : ", np.array(image_hsv))

# #########################################################################################################


#         contour_color = None
#         yellow_pixels = []
#         white_pixels = []
#         color_map = np.array(cut_image_hsv) 
#         #print("color map : ", color_map.shape)

#         for j,w in enumerate(color_map):
#             if (color_map[j][0][0] >= 25 and color_map[j][0][0] <= 35) and (color_map[j][0][1] >= 50 and color_map[j][0][1] <= 255) and (color_map[j][0][2] >= 70 and color_map[j][0][2] <= 255) :
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

# #         print("yellow : ", len(yellow_pixels))
# #         print('white : ', len(white_pixels))
        


#         #cv2.drawContours(coins , [cnt], -1, (0, 0, 255), -1)
#     #break

# cv2.imshow('coins', coins)
# cv2.waitKey(0)
# cv2.destroyAllWindows()








     


