import cv2
import numpy as np
import time
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#print(classes)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Load Video
#for real time:VideoCapture(0), for Pre loaded video:VideoCapture(images/alking.mp4)
cap = cv2.VideoCapture(0)#cap = cv2.VideoCapture("images/walking.mp4")
font =cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
	_, frame = cap.read()#we take 1st frame and execute the whole code below then move to 2nd frame and so on..
	frame_id += 1
	height, width, channels = frame.shape

	#Detecting Objects
	blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

	#for b in blob:
		#for n, img_blob in enumerate(b):
			#cv2.imshow(str(n), img_blob)

	net.setInput(blob)
	outs = net.forward(output_layers)

	#Showing informations on the Screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.2:#upon increasing we get less objects but detection is more accutare and vice- versa
				centre_x = int(detection[0] * width)
				centre_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				#mark centre of objects with circle
				#cv2.circle(img, (centre_x, centre_y), 10, (0, 255, 0), 2)

				#Rectangle Coordinates
				x = int(centre_x - w/2)#get top left X
				y = int(centre_y - h/2)

				#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	#print(len(boxes))
	#number_objects_detected = len(boxes)
	indexes =cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	#print(indexes)

	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			#print(label)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, label, (x, y+30), font, 2, (0, 0, 0), 2)


	elapsed_time = time.time() - starting_time
	fps = frame_id / elapsed_time
	cv2.putText(frame, "FPS:" + str(round(fps, 2)), (10, 30), font, 3, (0, 0, 0), 3)
	cv2.imshow("Image Window", frame)
	key = cv2.waitKey(1)#1 denotes that the loop waits for 1 milli second and again the loop continues
	if key == 13:
		break
cap.release()#closes the camera after we press the closing key
cv2.destroyAllWindows()
