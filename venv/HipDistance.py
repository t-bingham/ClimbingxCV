import cv2
import numpy as np
import time
import mediapipe as mp
import imutils
import PoseModule as pm

cap = cv2.VideoCapture('Media/1.mp4')
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = imutils.resize(img, width=400)
    img = detector.findPose(img=img, draw=False)
    lmList = detector.getPosition(img=img, draw=False)
    if len(lmList) != 0:
        detector.getAcuteAngle(img, 11, 23, 25)
        detector.getAcuteAngle(img, 12, 24, 26)
    cv2.imshow("Climbing", img)
    cv2.waitKey(1)
