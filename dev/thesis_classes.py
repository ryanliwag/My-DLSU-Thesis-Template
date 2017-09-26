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

#TCP_IP COMMUNICATION CLASS
# improvments pickle the image data
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
