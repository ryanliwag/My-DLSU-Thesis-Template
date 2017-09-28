
import numpy as np
import argparse
import imutils
import cv2
from math import sqrt

from thesis_classes import *

def capture(frame):
	try:
		X, Y, Z, box, midx, midy = get_size(frame, 90)
		#locates center of object
		cv2.circle(frame, (int(midx), int(midy)), 5, (0, 0, 255), -1)
		print(X, Y, Z)
		cv2.drawContours(frame,[box],0,(0,0,255),1)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame,"Parameters",(10,31), font, 0.5,(0,100,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Width:  {:.03}in".format(float(X)),(10,50), font, 0.5,(0,100,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Length: {:.03}in".format(float(Y)),(10,69), font, 0.5,(0,100,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Height: {:.03}in".format(float(Z)),(10,88), font, 0.5,(0,100,0),2,cv2.LINE_AA)

		cv2.imshow("processed image", frame)

	except IndexError:
		print("no contours to capture")
		pass


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def get_size(image_frame, calibrated_pxm):
	#This function returns Y dimension, X dimension, Area, midx, midy, 



	gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	#uncomment to see processed image
	#cv2.imshow('imclose', image)

	#Find contours
	(_,cnts,_) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#append contour into tha area array
	areaArray = []
	for i, c in enumerate(cnts):

		area = cv2.contourArea(c)
		#adjust this if possible
		areaArray.append(area)

	sorteddata = sorted(zip(areaArray,cnts), key=lambda x: x[0], reverse=True)


	#largest Contour
	c = sorteddata[0][1] 

	box = cv2.minAreaRect(c)
	box_points = cv2.boxPoints(box)
	box_points = np.int0(box_points)
	

	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	
	#extract 4 corners
	(tl, tr, br, bl) = box_points
	
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(midx, midy) = midpoint((tltrX, tltrY), (blbrX, blbrY))

	#compute distance x and y
	dA = sqrt( (tr[0] - tl[0])**2 + (tr[1] - tl[1])**2 )
	dB = sqrt( (tr[0] - br[0])**2 + (tr[1] - br[1])**2 )

	# X,Y,Z parameters
	X = dA / calibrated_pxm
	Y = dB / calibrated_pxm
	Z = get_arduino_data("/dev/ttyUSB0")

	return X, Y, Z, box_points, midx, midy


#Function to get height and weight of mango from arduino
def get_arduino_data(port):
	arduino_comms = arduino(port, 9600)
	height = arduino_comms.read_data()
	return height


def save_image(frame):
	time = datetime.datetime.now().time()
	cv2.imwrite(str(time) + ".jpeg", frame)
	print("Image saved" + str(time) + ".jpeg")


