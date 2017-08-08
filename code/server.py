import socket
import cv2
import numpy as np
import sys
import threading

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf



def main():
	TCP_IP = "192.168.1.107"
	TCP_PORT = 5000

	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind((TCP_IP, TCP_PORT))
	s.listen(1)
	while True:
		conn, addr = s.accept()

		try: 
			length = recvall(conn,16)
			stringData = recvall(conn, int(length))
			send_test = "received image"
			data = np.fromstring(stringData, dtype='uint8')
			decimg=cv2.imdecode(data,1)		
			conn.send(send_test.encode())
			cv2.imwrite('test.png',decimg)

		except:
			print("communication failed restart server")
			break

	cv2.destroyAllWindows() 
	s.close()



if __name__ == '__main__':
	main()