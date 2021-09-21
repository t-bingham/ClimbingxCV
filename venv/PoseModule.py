import cv2
import mediapipe as mp
import time
import imutils
import math


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                self.detectionCon, self.trackingCon)
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                      self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if (draw==True):
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return self.lmList

    def getAcuteAngle(self, img, p1, p2, p3, draw=True):

        #Get landmarks
        _, x1, y1 = self.lmList[p1]
        _, x2, y2 = self.lmList[p2]
        _, x3, y3 = self.lmList[p3]

        #Calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        #Correct angle
        angle = int(angle) % 360
        if (abs(int(angle)) > 180):
            angle = angle + (2 * (180 - angle))

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255) )
            cv2.circle(img, (x2, y2), 15, (0, 0, 255))
            cv2.circle(img, (x3, y3), 15, (0, 0, 255))
            cv2.putText(img, str(abs(int(angle))), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return angle





def main():
    cap = cv2.VideoCapture('Media/5.mp4')
    prevTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = imutils.resize(img, width=400)
        img = detector.findPose(img=img)
        lmList = detector.getPosition(img=img)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Climbing", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()