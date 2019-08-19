from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

filename='video12.mp4'

import cv2
cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        cv2.imshow('window-name',frame)
        cv2.imwrite("./out/frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10000):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

