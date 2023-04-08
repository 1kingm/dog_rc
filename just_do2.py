import cv2

from detect import double_bridge

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)
    a, b, c = double_bridge.double_bridge(frame)

