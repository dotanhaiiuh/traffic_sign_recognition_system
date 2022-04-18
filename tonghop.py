import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import numpy
#load the trained model to classify sign
from tensorflow.keras.models import load_model
model = load_model('static/model/traffic_classifier.h5')


classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }

def empty(a):
    pass
 
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
 
def getContours(img, imgContour, imgOr, *result, num):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area>800:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            # print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            
            k=12
 
            cv2.rectangle(imgContour,(x-k,y-k),(x+w+k,y+h+k),(0,255,0),3)

            pts1 = np.float32([[x-k, y-k], [x+w+k, y-k], [x-k, y+h+k], [x+w+k, y+h+k]]) 
            pts2 = np.float32([[0, 0], [30, 0], [0, 30], [30, 30]]) 
            matrix = cv2.getPerspectiveTransform(pts1, pts2) 
            result = cv2.warpPerspective(imgOr, matrix, (30, 30))
            result = numpy.expand_dims(result, axis=0)
            result = numpy.array(result)
            
            pred = model.predict_classes(result)[0]
            sign = classes[pred+1]
            cv2.putText(imgContour,sign,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)

def fillHoles(im_in):
    # Read image
    # im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE)
    # im_in = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)
    # th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_in.copy()

    h, w = im_in.shape[:2]

    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_in | im_floodfill_inv

    # cv2.imshow("Foreground", im_out)
    return im_out 

def imageProcess(img, imgContour, imgResult):
    imgBlur = cv2.GaussianBlur(imgResult,(5,5),0)
    imgCanny = cv2.Canny(imgBlur,150,100)
    # imgCanny = fillHoles(im_in)
    # kernel = np.ones((2, 2))
    # kernel2 = np.ones((3, 3))
    # imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgResult, imgContour, img, num=0)
    return imgCanny
            
frameWidth = 600
frameHeight = 600
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# path = 'static/img/camdi.jpg'
# path=filedialog.askopenfilename()
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",104,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",97,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
 
while True:
    success, img = cap.read()
    
    # img = cv2.imread(path)
    imgContour = img.copy()
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    # lowerR = np.array([0,135,135])
    # upperR = np.array([18,255,255])
    # lower = np.array([0, 91, 90])
    # upper = np.array([179, 255, 255])
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    maskfillHoles = fillHoles(mask)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgResult = cv2.bitwise_and(imgGray,imgGray,mask=maskfillHoles)


    imgDil = imageProcess(img, imgContour, imgResult)
    
    imgBlank = np.zeros_like(img)
    imgStack = stackImages(1,[imgContour])

    cv2.imshow("Stacked Images", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break