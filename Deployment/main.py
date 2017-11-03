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

from size_classifier import load_graph

class Import_Frcnn():
	def __init__(self, location):
		self.graph_frcnn = tf.Graph()
		self.sess = tf.Session(graph=self.graph_frcnn)
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
		self.sess = tf.Session(graph = self.graph_mtl)
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


def get_box(boxes, image):
	boxes = np.squeeze(boxes)
	box_to_color_map = collections.defaultdict(str)
	boxes_ = tuple(boxes[0].tolist())
	box_to_color_map[boxes_] = 'black'

	for box, color in box_to_color_map.items():
	    ymin, xmin, ymax, xmax = box
	    height, width = image.shape[:2]

	return xmin * width, xmax * width, ymin * height, ymax * height

 
def main():
	category_index = {1: {'id': 1, 'name': 'Green'}}
	ripeness_dict = {0: 'Green', 1: 'Semi-Ripe', 2: 'Ripe'}
	quality_dict = {0: 'Good', 1: 'Defect'}

	image_bgr = cv2.imread("/home/elements/Desktop/v-env/tensorflow/Dataset/ripeness/ripe/20171018-033104-604590.jpg")
	image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

	#import models
	model_fcnn = Import_Frcnn('/home/elements/Desktop/v-env/tensorflow/My-Tensorflow-Basics/object-detection/notebook/frozen_inference_graph.pb')
	model_mtl = Import_MTL("frozen_models/MTL_frozen_model.pb")


	(boxes, scores, classes, num) = model_fcnn.run(image)
	print("{0:.2f}%".format(scores.item(0) * 100))
	left, right, top, bottom = get_box(boxes, image)
	crop = image[int(top):int(bottom), int(left):int(right)]

	quality, ripeness = model_mtl.run(crop)
	print(quality_dict[np.argmax(quality, axis=1)[0]])
	print(ripeness_dict[np.argmax(ripeness, axis=1)[0]])

	cv2.imshow("cropped", crop)
	cv2.imshow("image", image_bgr)
	cv2.waitKey()




if __name__ == "__main__":
	main()




