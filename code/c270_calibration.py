import cv2
import time
def main():
	cap = cv2.VideoCapture(0)
	cap.set(3,1280)
	cap.set(4,720)

	time.sleep(2)
	cap.set(15, -0.5)




	while(True):
		ret, frame = cap.read()
		cv2.imshow('video', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()




if __name__ == "__main__":
	main()