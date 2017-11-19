
import argparse
import cv2
import numpy as np
import os
import random
import sys

from size_classifier import *
from mango_parameters import *





cap = cv2.VideoCapture(0)
if (cap.isOpened()):
	print("Camera OK")
else:
	cap.open()

size = ''

while (True):
	ret, original = cap.read()

	try:
		x, y = get_size(original,82)
		weight, size=predict_size([x,y])
	except:
		pass


	cv2.putText(original, "{}".format(size), (10,10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,255,0))
	cv2.imshow("ckass", original)


	if (cv2.waitKey(1) & 0xFF == ord('q')):xxxxxxxxxxxxxxxxxxxxxxxx
		break;

cap.release()
cv2.destroyAllWindows()
sys.exit()