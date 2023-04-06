import cv2
import serial
import numpy as np
import sys, math
from imutils import perspective

video_name = 1
com_name = 'COM3'
task_number = 0
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 膨胀腐蚀操作卷积核大小
threshold_dic = {'lower_seesaw' : np.array([7, 122, 48]), 'upper_seesaw' : np.array([33, 251, 145]), # 跷跷板
                 'lower_hinder_red' : np.array([7, 122, 48]), 'upper_hinder_red' : np.array([33, 251, 145]), # 跳高
                 'lower_hinder_white' : np.array([7, 122, 48]), 'upper_hinder_white' : np.array([33, 251, 145]), # 跳高
                 'lower_floor' : np.array([14, 71, 56]), 'upper_floor' : np.array([54, 121, 188]),  # 楼梯
                 'lower_dwb' : np.array([7, 122, 48]), 'upper_dwb' : np.array([33, 251, 145]), # 双木桥
                }

def Init_func():
    # 初始化串口
    mpucom = serial.Serial(port=com_name,  # 串口，请使用/dev/ttyUSB0作为移植代码
                        baudrate=115200,      # 波特率
                        bytesize=8,           # 数据位
                        stopbits=1,           # 停止位
                        timeout=0.1,          # 间隔
                        parity='N')           # 校验位
    if (mpucom.is_open):
        print("串口开启成功\r\n")
    else:
        raise SystemExit('Failed to open the serial port!')
    # 第二个初始化
    cap = cv2.VideoCapture(video_name)
    if cap.isOpened() == 0:
        print("Can't open the camera!")
    else:
        print("Camera is opening!")
        raise SystemExit('Failed to open the Camera!')
    return mpucom, cap

def about_equal(x, y, percent):
    if (x >= y*(1+percent)) & (x <= y*(1-percent)):
        return 1
    else:
        return 0
    
def front_seesaw(dot3, dot4):
    if about_equal(dot3[0], dot4[0], 0.05):
        y = dot3[1] + dot4[1]
        k = 0
    else:
        y = 0
        k = (dot3[1] - dot4[1])/(dot3[0] - dot4[0])
    return [y, k]

def on_seesaw(mask):
    img_canny = cv2.Canny(mask, 40, 100)
    lines = cv2.HoughLinesP(img_canny, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)
    list_kb = []
    x_median = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        k = (x1 - x2)/(y1 - y2)
        b = x1 - k*y1
        list_kb.append([k, b])
    for y in range(1, mask.shape[0]):
        x1 = list_kb[0][0]*y + list_kb[0][1]
        x2 = list_kb[1][0]*y + list_kb[1][1]
        x_median.append((x1+x2)/2)
    # 最小二乘法拟合
    mid_param = np.polyfit(x_median, range(1, mask.shape[0]), 1)
    start_point = [0, int(np.polyval(mid_param, 0))]
    end_point = [299, int(np.polyval(mid_param, 299))]
    try:
        return (start_point[1] - end_point[1])/(start_point[0] - end_point[0])  
    except:
        return 0

def Analyzing_conditions():
    pass

def find_seesaw(frame):
    global current_task
    y, k = 0, 0
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, threshold_dic['lower_seesaw'], threshold_dic['upper_seesaw'])
    mask_median = cv2.medianBlur(mask, 3)
    mask_open = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)
    mask_open = 255 - mask_open
    contours, _hierarchy = cv2.findContours(mask_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    try:
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        max_rect = cv2.minAreaRect(contours[max_id])  # 最小外接矩阵
        max_box = np.int0(cv2.boxPoints(max_rect)) # 四个顶点并取整数
        frame = cv2.drawContours(frame,contours[max_id],-1,(0,255,0),2) 
        frame = cv2.drawContours(frame,[max_box],0,(0,255,0),2) 
        
        # 最上面的两个点极其靠近图像最高点，则检测到了
        # 这里放一个赋值语句,需要更新计算方法，如果倾斜检测修改
        dot1, dot2, dot3, dot4 = perspective.order_points(max_box)
        if about_equal(dot1[0], dot2[0], 0.05) & about_equal(dot1[1], 0, 0.05) & about_equal(dot2[1], 0, 0.05):
            [y, k] = front_seesaw(dot3, dot4)
            print("front")
        elif ((not about_equal(dot1[0], dot2[0], 0.05)) & (not about_equal(dot1[1], 0, 0.05)) & (not about_equal(dot2[1], 0, 0.05))) & (about_equal(dot3[0], dot4[0], 0.05) & about_equal(dot3[1], frame.shape[0], 0.05) & about_equal(dot2[1], frame.shape[0], 0.05)):
            k = on_seesaw(mask_open)
            y = 0
            print("on")
        # TODO 添加一个判断条件
        elif about_equal(dot1[0], dot2[0], 0.05):
            current_task = current_task + 1
    except ValueError:
        print("未检测到轮廓")
    
    return y, k

def find_hinder(frame): # 跳高
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(frame_hsv, threshold_dic['lower_hinder_red'], threshold_dic['upper_hinder_red'])
    mask_white = cv2.inRange(frame_hsv, threshold_dic['lower_hinder_white'], threshold_dic['upper_hinder_white'])
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
        
def double_bridge(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, threshold_dic['lower_dwb'], threshold_dic['upper_dwb'])
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
            x_prime.append(
                
                np.polyval(param, y))  # 用y带入已知直线得到x_prime,两个轮廓
            
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

def find_floor(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, threshold_dic['lower_floor'], threshold_dic['upper_floor'])
    mask_median = cv2.medianBlur(mask, 3)
    mask_open = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)

    contours, _hierarchy = cv2.findContours(mask_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    
    
    for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))
    max_id = areas.index(max(areas))
    max_rect = cv2.minAreaRect(contours[max_id])  # 最小外接矩阵
    max_box = np.int0(cv2.boxPoints(max_rect)) # 四个顶点并取整数
    frame = cv2.drawContours(frame,contours[max_id],-1,(0,255,0),2)
    frame = cv2.drawContours(frame,[max_box],0,(0,255,0),2)
    
    approxs = cv2.approxPolyDP(contours[max_id], 10, True)  # 多边形逼近轮廓, 精度10像素
    cv2.polylines(frame, [approxs], True, (0, 0, 255), 3)
    # 这里放一个获取下面两坐标
    box = perspective.order_points(max_box)
    bot1 = box[2]
    bot2 = box[3]
    if about_equal(bot1[1], bot2[1], 0.01):
        # 前进
        if bot1[1] >= frame.shape[0] * 0.85:
        # 跳跃
            pass
    else:# 旋转
        k = (bot1[1] - bot2[1])/(bot1[0] - bot2[0])
        pass
    
judge_task = {
    0:find_seesaw,
    1:find_hinder,
    2:double_bridge,
    3:find_floor
}

if __name__ == '__main__':
    mpucom, cap = Init_func()
    current_task = 0
    while True:
        ret, frame = cap.read()
        judge_task[current_task](frame)