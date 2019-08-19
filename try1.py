# USAGE
# python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/adrian

# import the necessary packages

from __future__ import print_function
from sklearn.externals import joblib
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
from imutils.video import VideoStream
import argparse
import imutils
import mahotas
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
args = vars(ap.parse_args())

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	cv2.waitKey(5000) & 0xFF
 
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	
        p = os.path.sep.join([args["output"], "{}.png".format(
			str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total += 1
        
        break
	# if the `q` key was pressed, break from the loop
	#elif :


# do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()


digit1=0
m=os.path.join('models','svm.cpickle')
#model1=os.path.split("model/svm.cpcikle")
print(m)
# load the model
model = joblib.load(m)

# initialize the HOG descriptor
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), transform = True)

# load the image and convert it to grayscale
#loc=str(input("enter the image name:"))
#n=os.path.join('images','cellphone.png')
image = cv2.imread(p)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur the image, find edges, and then find contours along
# the edged regions
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort the contours by their x-axis position, ensuring
# that we read the numbers from left to right
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])

# loop over the contours
for (c, _) in cnts:
	# compute the bounding box for the rectangle
	(x, y, w, h) = cv2.boundingRect(c)

	# if the width is at least 7 pixels and the height
	# is at least 20 pixels, the contour is likely a digit
	if w >= 7 and h >= 20:
		# crop the ROI and then threshold the grayscale
		# ROI to reveal the digit
		roi = gray[y:y + h, x:x + w]
		thresh = roi.copy()
		T = mahotas.thresholding.otsu(roi)
		thresh[thresh > T] = 255
		thresh = cv2.bitwise_not(thresh)

		# deskew the image center its extent
		thresh = dataset.deskew(thresh, 20)
		thresh = dataset.center_extent(thresh, (20, 20))

		cv2.imshow("thresh", thresh)

		# extract features from the image and classify it
		hist = hog.describe(thresh)
		digit = model.predict([hist])[0]
                digit1=digit1+digit
                 
              
		print("I think that number is: {}".format(digit))
                

		# draw a rectangle around the digit, the show what the
		# digit was classified as
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.putText(image, str(digit), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
		cv2.imshow("image", image)
		cv2.waitKey(1000)
#print(hist)
print("final result is:{}".format(digit1))               
	
