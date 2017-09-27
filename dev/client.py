#made by Ryan Joshua Liwag


import numpy as np
import argparse
import imutils
import cv2
from math import sqrt

import serial 
import time
import sys
import struct

import vision
'''
To do list

1. Create A funciton that will receive all arduino data
2. Create A function that will transmit and receive all image and results through tcp_ip
3. Create A function that will transmit all necessary commands to Arduino
'''

# Main function
def main():
	#1 is for webcam another video capture will be here.
	cap = cv2.VideoCapture(0)
	cap.set(3,720)
	cap.set(4,720)
	while(True):

		ret, frame = cap.read()
		



		cv2.imshow('video', frame)

		#capture replace later with kivy button
		vision.capture(frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main() 
