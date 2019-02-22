# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# Substract shoots from target board image, return result image in GrayScale
def substract_shot(image,sens):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,255-sens], dtype=np.uint8)
    upper_white = np.array([255,sens,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image,image, mask= mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res
# Canny function with input filter arguments
def canny(image,c1,c2):
    blur = cv2.GaussianBlur(image,(5,5), 0)
    canny = cv2.Canny(blur,c1,c2)
    return canny
# Detect circle on image return prepeared image for detection, image with detected
# circles and coordinates and radius of each circle
def detect_circle(canny_image,image,minR,maxR):
    circle_image = cv2.cvtColor(canny_image,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(canny_image,cv2.HOUGH_GRADIENT,1,20,
                                param1=15,param2=4,minRadius = minR,maxRadius = maxR)
    try:
        circles = np.uint16(np.around(circles))
    except AttributeError:
        circles = np.array([])
        return (canny_image,image,circles)
    for i in circles[0,:]:
        detected_image = cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
        detected_image = cv2.circle(image,(i[0],i[1]),1,(0,0,255),3)
    return (circle_image,detected_image,circles)
# Calculate distance between two points
def distance(x0,y0,x1,y1):
    dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    return dist
# Calculate score and return sum score and vector with partial shoots score
def score(circlesXY,circles):
    # Get center coordinates of big black center circle and his radius
    # These are now references for calculating score
    x0 = int(circlesXY[0,:][0][0])
    y0 = int(circlesXY[0,:][0][1])
    r0 = int(circlesXY[0,:][0][2])
    # Score variable
    score = np.array([])
    # Victor with radius of target levels
    t_dis = [r0-24*3,r0-24*2,r0-24,r0,r0+24,r0+24*2,r0+24*3,r0+24*4,r0+24*5,r0+24*6]
    # If do not have any shot
    if circles.shape == (0,):
        score = []
        sum_score = 0
    else:
        # For each detected shot calculate distance and compare to target level
        for j in circles[0,:]:
            if distance(x0,y0,j[0],j[1]) < t_dis[0]:
                score = np.append(score,10)
            elif distance(x0,y0,j[0],j[1]) < t_dis[1]:
                score = np.append(score,9)
            elif distance(x0,y0,j[0],j[1]) < t_dis[2]:
                score = np.append(score,8)
            elif distance(x0,y0,j[0],j[1]) < t_dis[3]:
                score = np.append(score,7)
            elif distance(x0,y0,j[0],j[1]) < t_dis[4]:
                score = np.append(score,6)
            elif distance(x0,y0,j[0],j[1]) < t_dis[5]:
                score = np.append(score,5)
            elif distance(x0,y0,j[0],j[1]) < t_dis[6]:
                score = np.append(score,4)
            elif distance(x0,y0,j[0],j[1]) < t_dis[7]:
                score = np.append(score,3)
            elif distance(x0,y0,j[0],j[1]) < t_dis[8]:
                score = np.append(score,2)
            elif distance(x0,y0,j[0],j[1]) < t_dis[9]:
                score = np.append(score,1)
            else:
                score = np.append(score,0)
            if j[2] > 6:
                score = np.append(score,score[-1])
    # Sum score
    sum_score = sum(score)
    # Print score vector
    print('Your score is: ',score)
    # Print sum score
    print('Your sum score is: ',sum_score)
    return(score,sum_score)
# Main part:
# Import image
image = cv2.imread('TestIm1.jpg',1)
# Make image copy
try:
    copy_image1 = image.copy()
    copy_image2 = image.copy()
except AttributeError:
    print('Image not read.')
# Substract shoots from the target
shots = substract_shot(copy_image1,50)
# Canny filter on substracted shoots
canny_shots = canny(shots,300,600)
# Detect shoots on filtered image
detected_shot,detected_image,circles = detect_circle(canny_shots,copy_image1,1,15)
# Call canny filter to substract referent level circle 
canny_target = canny(copy_image2,300,500)
# Get center coordinates and radius of referent level circle
_,_,circlesXY =  detect_circle(canny_target,copy_image2,91,91)
# Get score and sum score
score,sum_score = score(circlesXY,circles)
# Show image with detected shoots
cv2.imshow('Target',detected_image)
# End of the program
