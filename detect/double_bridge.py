import cv2
import numpy as np
import math
from typing import List

video_name = 1
cap = cv2.VideoCapture(video_name)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 膨胀腐蚀操作卷积核大小
hsv_low = np.array([0,50,81])
hsv_high = np.array([50,91,255])

def length_line(x0, y0, x1, y1):
    length = math.sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1))
    return length

while True:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, hsv_low, hsv_high)
    mask_median = cv2.medianBlur(mask, 3)
    mask_open = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)
    
    # contours, _hierarchy = cv2.findContours(mask_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # areas = []
    
    # for c in range(len(contours)):
    #     areas.append(cv2.contourArea(contours[c]))
    # areas.sort(reverse = True)
    
    # for i in range(0, 1):
    frame_canny = cv2.Canny(mask_open, 20, 100)
    lines = cv2.HoughLinesP(frame_canny, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    length: List[float] = [0.001]
    
    for line in lines:
        x0, y0, x1, y1 = line[0]
        length.append(length_line(x0, y0, x1, y1))

    length_sort = length
    length_sort.sort(reverse=True)
    print(length_sort)
    # 0,1,2,3
    line0 = lines[length.index(length_sort[0])]
    line1 = lines[length.index(length_sort[1])]
    line2 = lines[length.index(length_sort[2])]
    line3 = lines[length.index(length_sort[3])]
    # x0, y0, x1, y1 = line1[0] 
    y = np.arange(frame.shape[0])
    x_median = []
    for i in y:
        x0 = line0[0][0] + (y - line0[0][1])*(line0[0][2] - line0[0][0])/(line0[0][3] - line0[0][1])
        x1 = line1[0][0] + (y - line1[0][1])*(line1[0][2] - line1[0][0])/(line1[0][3] - line1[0][1])
        x2 = line2[0][0] + (y - line2[0][1])*(line2[0][2] - line2[0][0])/(line2[0][3] - line2[0][1])
        x3 = line3[0][0] + (y - line3[0][1])*(line3[0][2] - line3[0][0])/(line3[0][3] - line3[0][1])
        x_median.append((x0 + x1 + x2 + x3)/4)
    print(len(x_median))
    print(len(y))
    mid_line = np.polyfit(x_median, y, 1)
    start_point = [0, int(np.polyval(mid_line, 0))]
    end_point = [frame.shape[1], int(np.polyval(mid_line, frame.shape[1]))]
    try:
        k = (start_point[1] - end_point[1])/(start_point[0] - end_point[0])  
    except:
        k = 0
    
        
