# import the necessary packages

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import face_recognition as fc

images = []
classNames = []
mylist = os.listdir(('images'))

for cls in mylist:
    curImg = cv2.imread(f'images/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])


print(classNames)
def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding complete')




def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


frame_counter = 0
face_recognition_interval = 1

# loop over the frames from the video stream
while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	if frame_counter % face_recognition_interval == 0:

		label = "Unknown"
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

		    # label = "Mask" if mask > withoutMask else "No Mask"
			
			if mask > withoutMask:
				label ="Mask"
				color = (0,255,0)

			else :
				img = frame
				imgs = cv2.resize(img , (0,0) , None , 0.25 , 0.25)
				imgs = cv2.cvtColor(imgs , cv2.COLOR_BGR2RGB)
				facesCurFrame = fc.face_locations(imgs)
				encodesCurFrame = fc.face_encodings(imgs , facesCurFrame )
        
				for encodeFace , faceLoc in zip(encodesCurFrame , facesCurFrame ):
					matches = fc.compare_faces(encodeListKnown , encodeFace)
					faceDis = fc.face_distance(encodeListKnown , encodeFace)
					print(faceDis)
					matchIndex = np.argmin(faceDis)

					if matches[matchIndex]:
						name = classNames[matchIndex].upper()
						print(name)
						label = name
						y1 , x2 ,y2 ,x1 = faceLoc
						x1 , x2 ,y1 ,y2 = x1*4 , x2*4 , y1*4 , y2*4
						cv2.rectangle(img , (x1,y1), (x2,y2) , (0,255,0) , 2)
						cv2.rectangle(img , (x1 , y2-35) , (x2,y2) , (0 ,255 ,0) , cv2.FILLED)
						cv2.putText(img ,name , (x1+6 , y2-6) , cv2.FONT_HERSHEY_COMPLEX ,1,(255 ,255 ,255) , 2)
							
				color = (255,0,0)


			if label == "Unknown" or label == "Mask":
			# cv2.rectangle(frame , (x1,y1), (x2,y2) , (0,255,0) , 2)
			# cv2.rectangle(frame , (x1 , y2-35) , (x2,y2) , (0 ,255 ,0) , cv2.FILLED)
			# cv2.putText(frame ,label , (x1+6 , y2-6) , cv2.FONT_HERSHEY_COMPLEX ,1,(255 ,255 ,255) , 2)
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255 , 255 ,255), 0)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("s"):
		break
	frame_counter+=1

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
