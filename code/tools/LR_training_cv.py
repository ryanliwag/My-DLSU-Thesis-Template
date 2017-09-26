#made by Ryan Joshua Liwag
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from math import sqrt

import serial 
import time
import sys
import struct

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
		self.data = self.ser.read(self.ser.inWaiting())
		self.data = self.data.decode("utf-8")
		return self.data


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Gets the length and Width of the Right most object in the screen
def get_length_width(image_frame, calibrated_pxm):

	#kernel
	kernel = np.ones((10,10), np.uint8)
	se = np.ones((10,10), dtype='uint8')
	er_kernel = np.ones((6,6), dtype='uint8')

	#Convert Image to grayscale
	image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

	#adaptive thereshold to determine contour
	image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 3)
	
	#morphology to remove noise
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

	#erode image
	image = cv2.erode(image, er_kernel, iterations = 1)

	#uncomment to see processed image
	#cv2.imshow('imclose', image)

	#Find contours
	(_,cnts,_) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#append contour into tha area array
	areaArray = []
	for i, c in enumerate(cnts):
		area = cv2.contourArea(c)
		areaArray.append(area)

	#sort the contour from largest to smallest
	sorteddata = sorted(zip(areaArray,cnts), key=lambda x: x[0], reverse=True)

	#largest Contour
	c = sorteddata[0][1] 


	box = cv2.minAreaRect(c)
	box_points = cv2.boxPoints(box)
	box_points = np.int0(box_points)
	yo=cv2.drawContours(image_frame,[box_points],0,(0,0,255),2)
	cv2.imshow('cool', yo)
	
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	'''
    leftmost = tuple(c[c[:,:,0].argmin()][0])
    rightmost = tuple(c[c[:,:,0].argmax()][0])
    topmost = tuple(c[c[:,:,1].argmin()][0])
    bottommost = tuple(c[c[:,:,1].argmax()][0])
    '''

    #pixel points
    1 mask = np.zeros(imgray.shape,np.uint8)
    2 cv2.drawContours(mask,[cnt],0,255,-1)
    3 pixelpoints = np.transpose(np.nonzero(mask))

    #orientation
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)

    #solidity
        1 area = cv2.contourArea(cnt)
    2 hull = cv2.convexHull(cnt)
    3 hull_area = cv2.contourArea(hull)
    4 solidity = float(area)/hull_area


	#extract 4 corners
	(tl, tr, br, bl) = box
	
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	print(tl)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(midx, midy) = midpoint((tltrX, tltrY), (blbrX, blbrY))

	#compute distance x and y
	dA = sqrt( (tr[0] - tl[0])**2 + (tr[1] - tl[1])**2 )
	dB = sqrt( (tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
	area = dA * dB
	dimA = dA / calibrated_pxm
	dimB = dB / calibrated_pxm
	area = dimA * dimB

	#return Y, X, Area, midx, midy
	return dimA, dimB, tr, dA, dB, midx, midy, c




#Function to get height and weight of mango from arduino
def get_arduino_data():
	arduino_comms = arduino("/dev/ttyUSB0", 9600)
	arduino_comms.send_data("height")
	height = arduino_comms.read_data()
	arduino_comms.send_data("weight")
	weight = arduino_comms.read_data()
	return height, weight


# Main function
def main():
	#1 is for webcam another video capture will be here.
	cap = cv2.VideoCapture(1)
	cap.set(3,1280)
	cap.set(4,720)
	while(True):

		ret, frame = cap.read()
		
		try:
			#145 calibrated for 11 inches camera height
			dimA, dimB, tl, dA, dB, midx, midy, cn = get_length_width(frame, 145)
			cv2.circle(frame, (int(midx), int(midy)), 5, (0, 0, 255), -1)
				#cv2.putText(frame, "{:.1}in".format(dimA), )
			font = cv2.FONT_HERSHEY_SIMPLEX
			area = dimA * dimB
			cv2.putText(frame,"Width:  {}in".format(float(dimA)),(10,50), font, 0.5,(0,100,0),2,cv2.LINE_AA)
			cv2.putText(frame,"Length: {}in".format(float(dimB)),(10,69), font, 0.5,(0,100,0),2,cv2.LINE_AA)
			cv2.putText(frame,"Area:   {}in".format(float(area)),(10,88), font, 0.5,(0,100,0),2,cv2.LINE_AA)

			
			cv2.drawContours(frame, cn, -1, (0, 255, 0), 2)
			print(midx, midy)
				#this might be useless unless mango is placed on the same spot
			cropped_img = frame[ tl[1]:tl[1]+int(dA), tl[0]:tl[0]+int(dB)]
			cv2.imshow('heyy', cropped_img)
		except:
			pass



	



		cv2.imshow('video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()






if __name__ == '__main__':
	main() 
