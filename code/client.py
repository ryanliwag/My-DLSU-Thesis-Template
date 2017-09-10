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

'''
To do list

1. Create A funciton that will receive all arduino data
2. Create A function that will transmit and receive all image and results through tcp_ip
3. Create A function that will transmit all necessary commands to Arduino
'''

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

'''
class tcp_ip():

	def __init__(self, TCP_IP, TCP_PORT):
		self.TCP_IP = TCP_IP
		self.TCP_PORT = TCP_PORT

		self.sock = socket.socket()
		self.sock.connect((self.TCP_IP, self.TCP_PORT))

	def send_image(self, frame):
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
		result, imgencode = cv2.imencode('.jpg', frame, encode_param)
		data = np.array(imgencode)
		stringData = data.tostring()
		self.sock.send( str(len(stringData)).ljust(16))
		self.sock.send(stringData)

	def receive_result(self):
		data = self.sock.recv(1024).decode()
		return data 
'''
'''
def send_image(frame):
	TCP_IP = "192.168.1.107"
	TCP_PORT = 5000

	sock = socket.socket()
	sock.connect((TCP_IP, TCP_PORT))

	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
	result, imgencode = cv2.imencode('.jpg', frame, encode_param)
	data = np.array(imgencode)
	stringData = data.tostring()

	sock.send( str(len(stringData)).ljust(16))
	sock.send(stringData)
	data = sock.recv(1024).decode()
	print(data)

'''


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Gets the length and Width of the Right most object in the screen
def get_length_width(image_frame, calibrated_pxm):
	#pixelsPerMetric = 90
	gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 4, 90, 90)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=3)
	edged = cv2.erode(edged, None, iterations=1)
	#cv2.imshow("gray",gray)
	#cv2.imshow("edged", edged)


	(_,cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#this needs to be calibrated to the distance of camera to the object
	c = max(cnts, key = cv2.contourArea)
	# if the contour is not sufficiently large, ignore it


	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")


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


def send_image(frame):
	TCP_IP = "192.168.1.107"
	TCP_PORT = 5000

	sock = socket.socket()
	sock.connect((TCP_IP, TCP_PORT))

	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
	result, imgencode = cv2.imencode('.jpg', frame, encode_param)
	data = np.array(imgencode)
	stringData = data.tostring()

	sock.send( str(len(stringData)).ljust(16))
	sock.send(stringData)
	data = sock.recv(1024).decode()
	print(data)


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
	while(True):

		ret, frame = cap.read()
		
		dimA, dimB, tl, dA, dB, midx, midy, cn = get_length_width(frame, 90)
		cv2.circle(frame, (int(midx), int(midy)), 5, (0, 0, 255), -1)
			#cv2.putText(frame, "{:.1}in".format(dimA), )
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame,"heigth: {:.01}in".format(float(dimA)),(10,50), font, 0.7,(255,255,255),2,cv2.LINE_AA)
		cv2.putText(frame,"length: {:.01}in".format(dimB),(10,67), font, 0.7,(255,255,255),2,cv2.LINE_AA)
		
		cv2.drawContours(frame, cn, -1, (0, 255, 0), 2)
		print(midx, midy)
			#this might be useless unless mango is placed on the same spot
		cropped_img = frame[ tl[1]:tl[1]+int(dA), tl[0]:tl[0]+int(dB)]
		cv2.imshow('heyy', cropped_img)


	



		cv2.imshow('video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()






if __name__ == '__main__':
	main() 
