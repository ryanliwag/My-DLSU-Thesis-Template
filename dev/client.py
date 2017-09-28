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

#IMPORT VISION FUNCTION
import vision

# Main function
def main():
	#1 is for webcam another video capture will be here.
	cap = cv2.VideoCapture(1)
	cap.set(3,720)
	cap.set(4,720)
	while(True):

		ret, frame = cap.read()

		cv2.imshow('video', frame)

		#capture replace later with kivy button
		if cv2.waitKey(1) & 0xFF == ord('c'):
			vision.capture(frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main() 
