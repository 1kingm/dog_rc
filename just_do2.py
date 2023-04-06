import cv2, math
import numpy as np
from imutils import perspective

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 膨胀腐蚀操作卷积核大小
hsv_low = np.array([0,50,81])
hsv_high = np.array([50,91,255])


def doubel_bridge(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, hsv_low, hsv_high)
    cv2.imshow('mask', mask)
    mask_median = cv2.medianBlur(mask, 3)
    cv2.imshow('mask_median', mask_median)
    mask_open = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)
    cv2.imshow('mask_open', mask_open)
    try:
        contours, _hierarchy = cv2.findContours(mask_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            frame = cv2.drawContours(frame,i,-1,(255,0,0),2)

        area = [-1, -2]
        for cnt in contours:
            area.append(cv2.contourArea(cnt))
        area_sort = area
        area_sort.sort()
        area1 = contours[contours.index(area_sort[-1])] 
        # 以下为计算距离
        rect1 = cv2.minAreaRect(area1)  # 最小外接矩阵
        area2 = contours[contours.index(area_sort[-2])]
        rect2 = cv2.minAreaRect(contours[area2])  # 最小外接矩阵
        box1 = np.int0(cv2.boxPoints(rect1)) # 四个顶点并取整数
        
        box2 = np.int0(cv2.boxPoints(rect2)) # 四个顶点并取整数
        box1_point = perspective.order_points(box1)
        box1_dot1 = box1_point[2]
        box1_dot2 = box1_point[3]
        length1 = math.sqrt(pow((box1_dot1[0]-box1_dot2[0]), 2) + pow((box1_dot1[1]-box1_dot2[1]), 2))
        
        rect2 = cv2.minAreaRect(contours[area2])  # 最小外接矩阵
        box2 = np.int0(cv2.boxPoints(rect2)) # 四个顶点并取整数
        box2_point = perspective.order_points(box2)
        box2_dot1 = box2_point[2]
        box2_dot2 = box2_point[3]
        length2 = math.sqrt(pow((box2_dot1[0]-box2_dot2[0]), 2) + pow((box2_dot1[1]-box2_dot2[1]), 2))
        
                
        y = np.array([100, 300])  #  [0-300]····
        x_prime = list()  # 列表
        k = 0
        for cnt in area1, area2:  # 可用轮廓
            x_coord = cnt[:, 0]  # 轮廓的x数值
            y_coord = cnt[:, 1]  # y
            param = np.polyfit(y_coord, x_coord, 1)  # 已有轮廓拟合直线
            x_prime.append(np.polyval(param, y))  # 用y带入已知直线得到x_prime,两个轮廓
            
        x_prime = np.array(x_prime)
        x_average = np.average(x_prime, axis=0)  # 所有轮廓x平均

        mid_param = np.polyfit(x_average, y, 1)
        start_point = [100, int(np.polyval(mid_param, 100))]
        end_point = [300, int(np.polyval(mid_param, 300))]
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        try:
            k = (start_point[1] - end_point[1])/(start_point[0] - end_point[0])  
        except:
            k = 0
            if length1 + length2 >= 40:
                k = 10000 # 可以跳跃
    except:
        print("contours")
        pass
            
    
    
    