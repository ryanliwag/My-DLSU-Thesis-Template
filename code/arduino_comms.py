import serial 
import time
import sys
import struct

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


if __name__ == '__main__':
    while(True):
        arduinoo = arduino("/dev/ttyUSB0", 9600)
        arduinoo.send_data("yowtf")
        data = arduinoo.read_data()
        print(data)
