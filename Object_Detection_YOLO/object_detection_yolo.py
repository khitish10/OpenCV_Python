import cv2
import numpy as np
#Load Yolo algo
#net=network,dnn=deep neural network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
###############################################################

#Load all files,read data and store inside classes[]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

layer_names = net.getLayerNames()#from net we get the layer names
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]#with output layer we get detection of objects

################################################################


#Load Image
img = cv2.imread("images/fruit.jpg")
#img = cv2.resize(img, None, fx=0.6, fy=0.6)#to crop image, fx=width, fy=height
height, width, channels = img.shape

#Detecting Objects, blob extracts features from image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)#scale factor=0.00392,size=(416 X 416),mean substraction from main layer=(0,0,0),True=change channel from BGR to RGB,crop=False i.e we do not want to change anything in image

#for b in blob:
    #for n, img_blob in enumerate(b):
        #cv2.imshow(str(n), img_blob)

net.setInput(blob)#pass blob image into the network
outs = net.forward(output_layers)
#print(outs)

################################################################

#Showing informations on the Screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:#we detect confidence in below 3 steps
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]#confidence=how confident the algo was in detecting objects
        if confidence > 0.2:#confidence in this case goes from 0-1, so adjust accordingly
            centre_x = int(detection[0] * width)
            centre_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            #mark centre of objects with circle
            #cv2.circle(img, (centre_x, centre_y), 10, (0, 255, 0), 2)#10=font size, 2=font thickness

            #Rectangle Coordinates
            x = int(centre_x - w/2)#get top left X
            y = int(centre_y - h/2)#get top left Y

            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)#square box around detected objects
            boxes.append([x, y, w, h])#we push rectangular areas into the objects
            confidences.append(float(confidence))#show % of confidence on screen
            class_ids.append(class_id)#to know the name of the object detected

#print(len(boxes))
#number_objects_detected = len(boxes)
indexes =cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#print(indexes)
print("\nObjects detected from Image are :")
font =cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]#get the coordinates of the rectangle
        label = str(classes[class_ids[i]])#name of the objects
        print(label)#display object names detected on display console
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)#show rectangle on objects detected
        cv2.putText(img, label, (x, y+30), font, 2, (0, 0, 0), 2)#put the text on detected objects



cv2.imshow("Image Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
