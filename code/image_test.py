#made by Ryan Joshua Liwag
#Coded with human music
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2



#to do list
#transfer of data from rpi to laptop
#weight sensor
#images measure size. this is where 2 cameras are needed. some correlation of these will be labeled
#large or small then fit it into a linear regression model.
#as for the 

# data transfer between python and rpi3
# multi-threading of images in rpi3


########################## ARDUINO COMMS #############################
import serial 
import time
import sys
import struct

class arduino():

	def __init__(self, usb_port, baud_rate):
		self.usb_port = usb_port
		self.baud_rate = baud_rate

		self.ser = serial.Serial(usb_port, baud_rate, timeout=0)

	def send_data(data):
		self.ser.write(str(data).encode())

	def read_data():
		data = self.ser.read(ser.inWaiting())
		data = data.decode("utf-8")
		return data



print("gg")
while True:

	ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0)


	one = 2

	#send data
	ser.write(str(one).encode())
	time.sleep(1)
	#receive data from arduino
	data=ser.read(ser.inWaiting())
	print(data.decode("utf-8"))



	



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
	dA = dist.euclidean(tl, tr)
	dB = dist.euclidean(tr, br)
	area = dA * dB
	dimA = dA / calibrated_pxm
	dimB = dB / calibrated_pxm
	area = dimA * dimB

	#return Y, X, Area, midx, midy
	return dimA, dimB, tr, dA, dB, midx, midy, c


# This function will calibrate the camera with the ultrasonic sensor
def calibrate():
	#calibrate with ultrasonic 
	print("calibration potangina mo, please die")


def feed_to_arduino():
	print("feed back data to arduino")

def get_height():
	print("get height info")

import serial 
import time
import sys
import struct

def test_arduino_comms():
	ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0)
	ser.write(bytes())
	print(ser.readline())

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


# Main function
def main():
	#1 is for webcam another video capture will be here.
	cap = cv2.VideoCapture(1)
	while(True):

		ret, frame = cap.read()
		try:
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
		except:
			print("not contours found")
			pass

		test_arduino_comms()
	
		if cv2.waitKey(1) & 0xFF == ord('c'):
			cv2.imwrite("captured.png", frame)
			send_image(frame)
		#this will be a ardiuno input



		cv2.imshow('video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

'''
down vote
	

i had this question and found another answer here: copy region of interest

If we consider (0,0) as top left corner of image called im with left-to-right as x direction and top-to-bottom as y direction. and we have (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region within that image, then:

roi = im[y1:y2, x1:x2]

'''





if __name__ == '__main__':
	main() 
