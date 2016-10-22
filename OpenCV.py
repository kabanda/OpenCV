import cv2
import numpy as np
from matplotlib import pyplot as plt

fileName = 'E:\MediaFire\Graphic Art\My Photos\Camera\IMAGE_037.jpg'
fileName2 = 'E:\MediaFire\Graphic Art\My Photos\Camera\\0001.png'
fileName3 = "E:\MediaFire\Graphic Art\My Photos\Camera\\testCircles1.jpg"
starsImage = "C:\Users\User\Documents\Visual Studio 2015\Projects\PythonApplication1\OpenCV\\stars.png"
starsImage2 = "C:\Users\User\Documents\Visual Studio 2015\Projects\PythonApplication1\OpenCV\\stars2.png"
starsImage3 = "C:\Users\User\Documents\Visual Studio 2015\Projects\PythonApplication1\OpenCV\\stars3.jpg"
starsImage4 = "C:\Users\User\Documents\Visual Studio 2015\Projects\PythonApplication1\OpenCV\\stars4.jpg"
starsImage5 = "C:\Users\User\Documents\Visual Studio 2015\Projects\PythonApplication1\OpenCV\\stars5.jpg"
check1 = "C:\Users\User\Documents\Visual Studio 2015\Projects\PythonApplication1\OpenCV\\Checkerboard_pattern.png"

def GetFile(fileName,size):
    img = cv2.imread(fileName)
    res = cv2.resize(img,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)
    return res


#img = cv2.imread("E:\MediaFire\Graphic Art\My Photos\Camera\IMAGE_037.jpg",cv2.IMREAD_GRAYSCALE)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#plt.imshow(img, cmap='gray', interpolation = 'bicubic')
#plt.plot([50,100],[200,1000], 'c', linewidth = 5)
#plt.show()

#########################camera################################


#cap = cv2.VideoCapture(0)
 
#while(True):
#    ret, frame = cap.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('gray',gray)

#    cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#cv2.destroyAllWindows()

#########################save camera ###############################

#cap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#while(True):
#    ret, frame = cap.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    out.write(frame)
#    cv2.imshow('frame',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#out.release()
#cv2.destroyAllWindows()

########################drawing and writing############################


#img = cv2.imread('E:\MediaFire\Graphic Art\My Photos\Camera\IMAGE_037.jpg',cv2.IMREAD_COLOR)
#cv2.line(img,(0,0),(200,300),(255,255,255),50)
#cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15)
#cv2.circle(img,(447,63), 63, (0,255,0), -1)
#pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
#pts = pts.reshape((-1,1,2))
#cv2.polylines(img, [pts], True, (0,255,255), 3)
#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,'OpenCV Tuts!',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

########################Image operataions############################

#img = cv2.imread('E:\MediaFire\Graphic Art\My Photos\Camera\IMAGE_037.jpg',cv2.IMREAD_COLOR)

##Region notation [yStart:yEnd,xStart,xEnd]
##img[500:700,500:700] = [255,255,255]

#copyFrom = img[500:700,500:700]

#img[200:400,150:350] = copyFrom

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

##########################Mouse events####################################
#events = [i for i in dir(cv2) if 'EVENT' in i]
#for e in events:
#   print e

## mouse callback function
#drawing = False # true if mouse is pressed
#mode = True # if True, draw rectangle. Press 'm' to toggle to curve
#ix,iy = -1,-1

## mouse callback function
#def draw_circle(event,x,y,flags,param):
#    global ix,iy,drawing,mode

#    if event == cv2.EVENT_LBUTTONDOWN:
#        drawing = True
#        ix,iy = x,y

#    elif event == cv2.EVENT_MOUSEMOVE:
#        if drawing == True:
#            if mode == True:
#                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#            else:
#                cv2.circle(img,(x,y),5,(0,0,255),-1)

#    elif event == cv2.EVENT_LBUTTONUP:
#        drawing = False
#        if mode == True:
#            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#        else:
#            cv2.circle(img,(x,y),5,(0,0,255),-1)

#img = np.zeros((512,512,3), np.uint8)
#cv2.namedWindow('image')
#cv2.setMouseCallback('image',draw_circle)

#while(1):
#    cv2.imshow('image',img)
#    k = cv2.waitKey(1) & 0xFF
#    if k == ord('m'):
#        mode = not mode
#    elif k == 27:
#        break

#cv2.destroyAllWindows()

#########################histogram###################################

#img = cv2.imread('E:\MediaFire\Graphic Art\My Photos\Camera\IMAGE_037.jpg',0)
#res = cv2.resize(img,None,fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC)

#cv2.imshow('original',res)

#plt.hist(res.ravel(),256,[0,256]);
#plt.show()

#################################
## Read image
#im = cv2.imread(starsImage5, cv2.IMREAD_GRAYSCALE)
#imfiltered = cv2.inRange(im,100,255)

#cv2.imshow("imfiltered", imfiltered)

##OPENING
#kernel = np.ones((1,1))

#opening = cv2.morphologyEx(imfiltered,cv2.MORPH_OPEN,kernel)

##write out the filtered image

#cv2.imwrite('colorfiltered.jpg',opening)


## Setup SimpleBlobDetector parameters.
#params = cv2.SimpleBlobDetector_Params()

#params.blobColor= 255
#params.filterByColor = True

## Filter by Area.
#params.filterByArea = True;
#params.minArea = .01;
#params.maxArea = 150000;

## Filter by Ci rcularity
#params.filterByCircularity = False
#params.minCircularity = 0.01

## Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.01
 
## Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.001

## Create a detector with the parameters
#ver = (cv2.__version__).split('.')
#if int(ver[0]) < 3 :
#    detector = cv2.SimpleBlobDetector(params)
#else : 
#    detector = cv2.SimpleBlobDetector_create(params)


## Detect blobs.
#keypoints = detector.detect(opening)

#print "Keypoint count = %d" % (len(keypoints))

## Draw detected blobs as green circles.
## cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
## the size of the circle corresponds to the size of blob

##print str(keypoints)

#im_with_keypoints = cv2.drawKeypoints(opening, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
##im_with_keypoints = cv2.drawKeypoints(opening, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
## Show blobs
#cv2.imshow("Keypoints", im_with_keypoints)
#img2gray = cv2.cvtColor(im_with_keypoints,cv2.COLOR_BGR2GRAY)
#ret,grayed = cv2.threshold(img2gray,12,255,cv2.THRESH_TOZERO)
#anded = cv2.bitwise_xor(imfiltered,grayed)
#cv2.imshow("anded", anded)
##cv2.imwrite('Keypoints.jpg',im_with_keypoints)

#img = cv2.imread(check1)img = GetFile(check1,.25)gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)corners = np.int0(corners)
for i in corners:    x,y = i.ravel()    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow("corners", img)
#plt.imshow(img)
#plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()