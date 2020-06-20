import cv2
import numpy as np
img=cv2.imread('images/color_code.jpg')#import file
while True:
    hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#convert RGB to HSV

    #Red Color
    low_red=np.array([161, 155, 84])#lowe range for Red Color
    high_red=np.array([179, 255, 255])#upper range for Red Color
    red_mask=cv2.inRange(hsv_img, low_red, high_red)#create mask
    red =cv2.bitwise_and(img, img, mask=red_mask)#extract image using bitwise
    cv2.imshow('Image', img)#show original image
    cv2.imshow('Red Mask', red)#show extracted image

    #Blue
    low_blue=np.array([94, 80, 2])
    high_blue=np.array([126, 255, 255])
    blue_mask=cv2.inRange(hsv_img, low_blue, high_blue)
    blue =cv2.bitwise_and(img, img, mask=blue_mask)
    cv2.imshow('Image', img)
    cv2.imshow('Blue Mask', blue)

    #Green
    low_green=np.array([25, 52, 72])
    high_green=np.array([102, 255, 255])
    green_mask=cv2.inRange(hsv_img, low_green, high_green)
    green =cv2.bitwise_and(img, img, mask=green_mask)
    cv2.imshow('Image', img)
    cv2.imshow('Green Mask', green)

    #All colors except white
    low=np.array([0, 42, 0])
    high=np.array([179, 255, 255])
    mask=cv2.inRange(hsv_img, low, high)
    result =cv2.bitwise_and(img, img, mask=mask)
    
    cv2.imshow('Frame', img)
    cv2.imshow('Detected Frame', result)

   
    
    key=cv2.waitKey(1)
    if key == 13:
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
