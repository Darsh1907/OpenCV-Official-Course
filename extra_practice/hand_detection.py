import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
Hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
pTime = 0

while True:
    success, img = cap.read()
    cTime = time.time()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int (lm.y*h)
                cv2.circle(img, (cx, cy), 2, (255, 0, 0), 2)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
    fps = int(1/(cTime-pTime))
    pTime = cTime
    cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
