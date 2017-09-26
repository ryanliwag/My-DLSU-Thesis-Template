import cv2 
import datetime

print(datetime.datetime.now())


def save_image(frame):
	time = datetime.datetime.now().time()
	cv2.imwrite(str(time) + ".jpeg", frame)




def main():
	cap = cv2.VideoCapture(0)
	while(True):
		ret, frame = cap.read()
		cv2.imshow('video', frame)


		if cv2.waitKey(1) & 0xFF == ord('c'):
			save_image(frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()




if __name__ == '__main__':
	main()