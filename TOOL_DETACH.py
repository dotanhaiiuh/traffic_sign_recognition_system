import cv2
import numpy as np
import matplotlib.pyplot as plt

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
 
def getContours(img, imgContour, imgOr, num, r):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area>500:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            # print(len(approx))
            objCor = len(approx)
            
            x, y, w, h = cv2.boundingRect(cnt)
 
            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"
 
            cv2.rectangle(imgContour,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
            
            # Locate points of the documents or object which you want to transform 
            # pts1 = np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]]) 
            # pts2 = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]]) 
            # # Apply Perspective Transform Algorithm
            # matrix = cv2.getPerspectiveTransform(pts1, pts2) 
            # result = cv2.warpPerspective(imgOr, matrix, (100, 100))
            if cv2.waitKey(1) & 0xFF == ord('t'):
                takePhoto(contours, imgOr, r, num)
            # r=r+1
            # cv2.imwrite(f'Test_gray_{r}-{num}.jpg', result) 
            

            # cv2.putText(imgContour,objectType,
            #             (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
            #             (0,0,0),2)
def takePhoto(contours, imgOr, r, num):
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
            k=2
            pts1 = np.float32([[x-k, y-k], [x+w+k, y-k], [x-k, y+h+k], [x+w+k, y+h+k]]) 
            pts2 = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]]) 
            # Apply Perspective Transform Algorithm
            matrix = cv2.getPerspectiveTransform(pts1, pts2) 
            result = cv2.warpPerspective(imgOr, matrix, (100, 100))
            r=r+1
            # cv2.imwrite(f'NewTrain/0/Test_gray_{r}-{num}.jpg', result) 
            cv2.imwrite(f'2_03.jpg', result)
    
def fillHoles(im_in):
    # Read imageq
    # im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE)
    # im_in = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)
    # th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_in.copy()

    h, w = im_in.shape[:2]

    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_in | im_floodfill_inv

    cv2.imshow("Foreground", im_out)
    return im_out 

def imageProcess(img, imgContour, imgResult):
    imgBlur = cv2.GaussianBlur(imgResult,(5,5),0)
    imgCanny = cv2.Canny(imgResult,150,100)
    # kernel = np.ones((2, 2))
    # # kernel2 = np.ones((3, 3))
    # imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgCanny, imgContour, img, num=0, r=0)
    return imgCanny

# path = 'Train/0/00000_00001_00001.png'
from tkinter import filedialog
path=filedialog.askopenfilename()
       
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",135,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",135,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
 
while True:
    img = cv2.imread(path)
    imgContour = img.copy()
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])

    mask = cv2.inRange(imgHSV,lower,upper)
    maskfillHoles = fillHoles(mask)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img.shape)
    print(imgGray.shape)
    imgResult = cv2.bitwise_and(img,img,mask=maskfillHoles)


    imgDil = imageProcess(img, imgContour, imgResult)
    


    imgBlank = np.zeros_like(img)
    imgStack = stackImages(0.4,([img, imgHSV, imgGray],
                                [imgResult,imgContour, imgDil]))

    cv2.imshow("Stacked Images", imgStack)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break