import cv2
import numpy as np
import math
from typing import List
from imutils import perspective

video_name = 1
cap = cv2.VideoCapture('1.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 膨胀腐蚀操作卷积核大小
hsv_low = np.array([23,154,180])
hsv_high = np.array([32,255,255])

def length_line(dot1, dot2):
    length = math.sqrt((dot1[0]-dot2[0])*(dot1[0]-dot2[0]) + (dot1[1]-dot2[1])*(dot1[1]-dot2[1]))
    return length


def nei_abf(dot1, dot2): # 计算斜率和截距
    Nei = (dot1[0] - dot2[0])/(dot1[1] - dot2[1]) # 斜率,
    abf = dot1[0] - Nei*dot1[1] # 截距
    return [Nei, abf] # x/y 默认斜率算法
    

def rect_dot(frame, area1, area2, number): 
    rect1 = cv2.minAreaRect(area1)
    rect2 = cv2.minAreaRect(area2)
    
    conflag1, conflag2 = 0.01, 0.01
    img_ele1, img_ele2  = conflag1 * cv2.arcLength(area1, True), conflag2 * cv2.arcLength(area2, True)
    approxCourve1= cv2.approxPolyDP(area1,img_ele1,closed = True)
    approxCourve2= cv2.approxPolyDP(area2,img_ele2,closed = True)
    while True: # 检测多边形为小于等于四条边的
        if len(approxCourve1) <= 4:
            break
        else:
            conflag1 = conflag1*1.5
            img_ele1 = conflag1 * cv2.arcLength(area1, True)
            approxCourve1= cv2.approxPolyDP(area1,img_ele1,closed = True)
    while True:
        if len(approxCourve2) <= 4:
            break
        else:
            conflag2 = conflag2*1.5
            img_ele2 = conflag2 * cv2.arcLength(area1, True)
            approxCourve2= cv2.approxPolyDP(area2,img_ele2,closed = True)
    
    
    approxCourve1 = np.array(approxCourve1)
    approxCourve1 = np.squeeze(approxCourve1, axis=1)
    approxCourve2 = np.array(approxCourve2)
    approxCourve2 = np.squeeze(approxCourve2, axis=1)
    
    # cv2.line(frame, approxCourve2[2], approxCourve2[1], (0, 255, 255), 2)
    dot_num1, dot_num2 =len(approxCourve1), len(approxCourve2)
    if rect1[0][0] > rect2[0][0]: # 判断两个矩形位置，根据位置确定取的直线, rect1是右边的，rect2是左边的
        if dot_num1 == 2 | dot_num2 == 2:
            if number == 1:
                return 0
            elif number == 2:
                return [0, 0], [0, 0]
        else:
            if number == 1:
                return length_line(approxCourve1[2], approxCourve1[dot_num1-1]) + length_line(approxCourve2[2], approxCourve2[1])
            elif number == 2:
                return nei_abf(approxCourve1[0], approxCourve1[1]), nei_abf(approxCourve2[0], approxCourve2[dot_num2-1])
    else:
        if dot_num1 == 2 | dot_num2 == 2:
            if number == 1:
                return 0
            elif number == 2:
                return [0, 0], [0, 0]
        else:
            if number == 1:
                return length_line(approxCourve1[2], approxCourve1[1]) + length_line(approxCourve2[2], approxCourve2[dot_num2-1])
            elif number == 2:
                return nei_abf(approxCourve1[0], approxCourve1[dot_num1-1]), nei_abf(approxCourve2[0], approxCourve2[1])

while cap.isOpened():
    ret, frame = cap.read()

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, hsv_low, hsv_high)
    
    mask_median = cv2.medianBlur(mask, 3)
    mask_open = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)


    contours, _hierarchy = cv2.findContours(mask_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 轮廓查找

    cnts = [] # 剔除部分之后的全部轮廓
    for cnt in range(len(contours)): # 剔除部分轮廓
        (x, y, w, h) = cv2.boundingRect(contours[cnt])
        if w > 10 and h > 30:
            cnts.append(contours[cnt])
    
    
    area = [] # 轮廓面积
    for cnt in cnts:
        area.append(cv2.contourArea(cnt))
        
    # area为所有的cnt的面积

    area_sort = area
    area_sort.sort(reverse=True) # 根据面积序降的area，排序结果
    try:
        if len(cnts) < 2:
            continue
        print(len(cnts))
        area1 = cnts[area.index(area_sort[0])] # 最大的轮廓
        area2 = cnts[area.index(area_sort[1])] # 第二大的轮廓


        [nei1, abf1], [nei2, abf2] = rect_dot(frame, area1=area1, area2=area2, number=2)
        if [nei1, abf1] == [0, 0] and [nei2, abf2] == [0, 0]:
            pass
        #TODO 根据实际图像尺寸修改
        y_coors = np.arange(1, 1080)
        x_coors = []

        for y in y_coors:
            x_coor = (y*nei1+abf1 +y*nei2+abf2)/2 
            x_coors.append(x_coor)
            cv2.circle(frame,(int(x_coor),int(y)),2,(0,0,255),-1)
        
        nei_median, _ = np.polyfit(y_coors, x_coors, 1) # 拟合直线
        print(nei_median)
        if (nei_median<0.1) and (nei_median>-0.1): #TODO 判断直行的条件
            print("无需转向")
            if rect_dot(frame, area1, area2, 1) >= 40: #TODO根据最下的判断是否可以跳跃
                print("跳跃")
            else:
                print("直行")
        else:
            k_median = 1/nei_median # 斜率y/x
            print("转向")
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    except:
        pass
