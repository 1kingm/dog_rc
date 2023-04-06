import cv2
import numpy as np

video_name = 1
red_hsv_high = np.array([255, 255, 244])
red_hsv_low = np.array([167, 50, 47])
white_hsv_low = np.array([0, 0, 186])
white_hsv_high = np.array([88, 145, 255])
cap = cv2.VideoCapture(video_name)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 膨胀腐蚀操作卷积核大小
while True:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(frame_hsv, red_hsv_low, red_hsv_high)
    mask_white = cv2.inRange(frame_hsv, white_hsv_low, white_hsv_high)
    mask = cv2.bitwise_or(mask_red, mask_white)
    mask_median = cv2.medianBlur(mask, 3)
    contours, _hierarchy = cv2.findContours(mask_median.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    try:
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        max_rect = cv2.minAreaRect(contours[max_id])  # 最小外接矩阵
        max_box = np.int0(cv2.boxPoints(max_rect)) # 四个顶点并取整数
        img2 = cv2.drawContours(frame,[max_box],0,(0,255,0),2) 
    except ValueError:
        print("未检测到轮廓")
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
    
