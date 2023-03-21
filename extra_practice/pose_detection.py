import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
cap = cv2.VideoCapture("PoseVideo.mp4")
Pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Pose.process(imgRGB)
    img = cv2.resize(img, (720, 405))
    if results.pose_landmarks:
        for id, lms in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lms.x*w), int(lms.y*h)
            print(id, cx, cy)
            cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    cv2.imshow("Image", img)
    cv2.waitKey(1)