import cv2
import numpy as np 
import socket

class tcp_ip():

	def __init__(self, TCP_IP, TCP_PORT):
		self.TCP_IP = TCP_IP
		self.TCP_PORT = TCP_PORT

		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.connect((self.TCP_IP, self.TCP_PORT))

	def send_image(self, frame):
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
		result, imgencode = cv2.imencode('.jpg', frame, encode_param)
		data = np.array(imgencode)
		stringData = data.tostring()
		self.sock.send((str(len(stringData)).ljust(16)).encode())
		self.sock.send(stringData)

	def receive_result(self):
		data = self.sock.recv(1024).decode()
		return data 


def send_image(frame):
	TCP_IP = "192.168.1.108"
	TCP_PORT = 5000

	sock = socket.socket()
	sock.connect((TCP_IP, TCP_PORT))

	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
	result, imgencode = cv2.imencode('.jpg', frame, encode_param)
	data = np.array(imgencode)
	stringData = data.tostring()

	sock.send( (str(len(stringData)).ljust(16)).encode())
	sock.send(stringData)
	data = sock.recv(1024).decode()
	print(data)




def main():
	cap = cv2.VideoCapture(0)


	while(True):
		ret, frame = cap.read()
		cv2.imshow('video', frame)


		if cv2.waitKey(1) & 0xFF == ord('c'):
			send_image(frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()




if __name__ == '__main__':
	main()

