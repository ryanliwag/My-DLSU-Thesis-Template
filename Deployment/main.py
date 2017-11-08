# step 1 press start run
# step 2 get arduino and size, width and length
# step 3 run motor for 1 sec and capture 4 frames
# step 4 run frames through frcnn, crop output then feed to MTL
# display result in opencv for now. 

'''
arduino_comms = mango_parameters.arduino("/dev/ttyUSB0", 9600)
arduino_comms.send_data("height")
height = arduino_comms.recv_data
'''
import tensorflow as tf
import cv2
import numpy as np
import collections
import time
from mango_parameters import *


import threading

configd = tf.ConfigProto()
configd.gpu_options.allow_growth=True



class Import_Frcnn():
	def __init__(self, location):
		self.graph_frcnn = tf.Graph()
		self.sess = tf.Session(graph=self.graph_frcnn, config=configd)
		with self.graph_frcnn.as_default():
			self.od_graph_def = tf.GraphDef()	
			with tf.gfile.GFile(location, 'rb') as fid:
				serialized_graph = fid.read()
				self.od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(self.od_graph_def, name='')

		self.image_tensor = self.graph_frcnn.get_tensor_by_name('image_tensor:0')
		self.detection_boxes = self.graph_frcnn.get_tensor_by_name('detection_boxes:0')
		self.detection_scores = self.graph_frcnn.get_tensor_by_name('detection_scores:0')
		self.detection_classes = self.graph_frcnn.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.graph_frcnn.get_tensor_by_name('num_detections:0')
		print("Model frcnn ready")

	def run(self, frame):
		image_np = np.expand_dims(frame, axis = 0)
		return self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],feed_dict={self.image_tensor: image_np})


class Import_MTL():

	def __init__(self, location):

		self.graph_mtl = load_graph(location)
		self.sess = tf.Session(graph = self.graph_mtl, config=configd)
		'''
		for op in self.graph_mtl.get_operations():
			print(str(op.name)) 
		'''
		self.y_pred_quality = self.graph_mtl.get_tensor_by_name("prefix/y_pred_quality:0")
		self.y_pred_ripeness = self.graph_mtl.get_tensor_by_name("prefix/y_pred_ripeness:0")
		self.x = self.graph_mtl.get_tensor_by_name("prefix/x:0") 
		print("Model MTL ready")


	def run(self, frame):
		image_np = self.predict(frame)
		return self.sess.run([self.y_pred_quality, self.y_pred_ripeness], feed_dict={self.x: image_np})


	def predict(self, image):
		image_rgb = cv2.resize(image, (50,50))
		image_rgb = np.expand_dims(image_rgb, axis = 0)
		return image_rgb


def get_box(boxes, scores, image):
	boxes = np.squeeze(boxes)
	boxes_array = []
	scores_array = []
	height, width = image.shape[:2]
	for i in range(3):
		if scores.item(i) > 0.8:
			ymin, xmin, ymax, xmax = boxes[i]
			boxes_array.append([xmin * width, xmax * width, ymin * height, ymax * height])
			scores_array.append(scores.item(i))
		else: 
			pass

	return boxes_array, scores_array





class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.stop = threading.Event()

		#predict up to 3 items only
		self.model_mtl = Import_MTL("frozen_models/MTL_frozen_model.pb")
		self.model_fcnn = Import_Frcnn('frozen_models/frozen_inference_graph.pb')

	def run(self, frame):
		# Load the VGG16 network
		return self.predict(frame)

	def predict(self, frame):
		ripe_score = []
		quality_score = []
		(boxes, scores, classes, num) = self.model_fcnn.run(frame)
		box_array, scores_ = get_box(boxes, scores, frame)

		if box_array:
			for g in range(len(scores)):
				left, right, top, bottom  = box_array[g]
				crop = frame[int(top):int(bottom), int(left):int(right)]
				quality, ripeness = self.model_mtl.run(crop)
				ripe_score.append(ripeness)
				quality_score.append(quality)

		return box_array, scores_, ripe_score, quality_score

		

	def terminate(self):
		self.stop.set()



def draw_boxes_scores(box_array, score_array, ripe_array, quality_array, frame):
	ripeness_dict = {0: 'Green', 1: 'Semi-Ripe', 2: 'Ripe'}
	quality_dict = {0: 'Good', 1: 'Defect'}
	for i in range(len(box_array)):
		try:
			cv2.rectangle(frame, (int(box_array[i][0]), int(box_array[i][2])), (int(box_array[i][1]), int(box_array[i][3])),(0,255,0),3)
			cv2.putText(frame, "Detection:{0:.2f}".format(score_array[i]), (int(box_array[i][0]),int(box_array[i][2]-6)), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,255,0))
			cv2.putText(frame, "Quality:{}".format(quality_dict[int(np.argmax(quality_array[i], axis=1))]), (int(box_array[i][0]),int(box_array[i][2]-17)), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,255,0))
			cv2.putText(frame, "Ripeness:{}".format(ripeness_dict[int(np.argmax(ripe_array[i], axis=1))]), (int(box_array[i][0]),int(box_array[i][2]-28)), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,255,0))
		except:
			pass


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def normalize_size(x):
    #Values obtained from train.py output
    mean = np.asarray([[76.06636364, 119.57220779]])
    std = np.asarray([[5.95719927, 8.19216614]])
    x_normalized = (x - mean) / std
    return x_normalized

def convert_sizes(size):
	size = int(size)
	if size >= 400:
		return "Large"
	elif size <= 399 and size >= 200:
		return "medium"
	elif size < 199:
		return "small"




def main():

	cap = cv2.VideoCapture(0)
	cap.set(3,1080)
	cap.set(4,720)
	if (cap.isOpened()):
		print("Camera OK")
	else:
		cap.open()

	yo_thread = MyThread()

	while(True):

		ret, frame = cap.read()
		frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		box, score, ripe, quality = yo_thread.run(frame_)



		draw_boxes_scores(box, score, ripe, quality, frame)

		cv2.imshow("Classification", frame)
		cv2.namedWindow('Classification',cv2.WINDOW_NORMAL)
		if (cv2.waitKey(1) & 0xFF == ord('q')):
			break;

	cap.release()
	frame = None
	cv2.destroyAllWindows()
	yo_thread.terminate()
	sys.exit()



if __name__ == "__main__":
	main()




