import serial 
import time
import sys
import struct

import numpy as np
import argparse
import cv2
from math import sqrt

#Arduino Communication Class
class arduino():

	def __init__(self, usb_port, baud_rate):
		self.usb_port = usb_port
		self.baud_rate = baud_rate

		self.ser = serial.Serial(usb_port, baud_rate, timeout=5)

	def send_data(self, data):
		self.ser.write(str(data).encode())
		time.sleep(0.5)

	def read_data(self):
		self.data = self.ser.readline().strip()
		#self.data = self.data.decode("utf-8")
		return self.data

	def close(self):
		self.ser.close()


def get_arduino_data(port):
	arduino_comms = arduino(port, 9600)
	height = arduino_comms.read_data()
	return height

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def get_size(image_frame, calibrated_pxm):
	#This function returns Y dimension, X dimension, Area, midx, midy, 

	gray = cv2.cvtColor(image_frame, 
						cv2.COLOR_BGR2GRAY)
	cv2.imwrite("gray.png", gray)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	cv2.imwrite("gaussianblur.png", gray)
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	canny = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(canny, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
	#cv2.imwrite("edged.png", edged)	
	#uncomment to see processed image
	#cv2.imwrite("gray_1.png", image_frame)
	#cv2.imwrite("canny.png", canny)
	#Find contours
	(_,cnts,_) = cv2.findContours(edged, 
								 cv2.RETR_EXTERNAL, 
								 cv2.CHAIN_APPROX_SIMPLE)

	#append contour into tha area array
	areaArray = []
	for i, c in enumerate(cnts):
		area = cv2.contourArea(c)
		#adjust this if possible
		areaArray.append(area)

	sorteddata = sorted(zip(areaArray,cnts), key=lambda x: x[0], reverse=True)

	#largest Contour
	c = sorteddata[0][1] 
	hey = cv2.drawContours(image_frame, c, -1, (255, 255, 0), 1)
	x,y,w,h = cv2.boundingRect(c)
	
	#cv2.imwrite("box1.png", hey)
	yo=cv2.rectangle(image_frame,(x,y),(x+w,y+h),(0,255,0),2)
	#cv2.imwrite("box.png", yo)


	#compute distance x and y
	dA = w
	dB = h

	# X,Y,Z parameters
	X = dA / calibrated_pxm
	Y = dB / calibrated_pxm
	Z = get_arduino_data("/dev/ttyUSB0")



	return X, Y, Z

'''
img = cv2.imread("test_images/20171018-032922-433214.jpg")
print(get_size(img, 90))
'''