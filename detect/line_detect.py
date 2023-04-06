import cv2
import numpy as np
import math
import serial
cap = cv2.VideoCapture('red_video.MP4')
kernel = np.ones((3, 3), dtype=np.uint8)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
width = 300
height = 300
lower_red = np.array([0, 62, 0])
upper_red = np.array([255, 255, 255])
minLineLength = 10
maxLineGap = 80
global send_data
send_data = []
# mpucom = serial.Serial(port='/dev/ttyUSB1',  # 串口，请使用/dev/ttyTHS1作为移植代码
#                        baudrate=115200,      # 波特率
#                        bytesize=8,           # 数据位
#                        stopbits=1,           # 停止位
#                        timeout=0.1,          # 间隔
#                        parity='N')           # 校验位

def send_com():
    if (np.array(send_data)>0).all() | (np.array(send_data)>0).all():
        # mpucom.write(sum(send_data)/len(send_data))
        print('')
    send_data.clear()
    
def Init_vary():
    global draw_number
    global slopes
    global coors
    global lengths
    
    draw_number = 0
    slopes = []
    coors = []
    lengths = []

def vague(slopes, slope):
    number = 0
    for equal in slopes:
        if 0.9 < slope/equal <1.1:
            number = number + 1 
            return 1, number
        number = number + 1
    return 0, 0
    
def Image_process(img):
    img_Gau = cv2.GaussianBlur(img, (3, 3), 0)  # 0: sigmaX=sigmaY
    img_hsv = cv2.cvtColor(img_Gau, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_red, upper_red)
    opening = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, 1)
    return opening 
   
def reduce_duplication(lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = (x1 - x2)^2 + (y1 - y2)^2
        coor = [x1, y1, x2, y2]
        if x1 == x2:
            slope = 90
        else :
            slope = math.atan((y1 - y2)/(x1 - x2))        
        just, number = vague(slopes, slope)
        if just:
            if length > lengths[number - 1]:
               slopes[number - 1] = slope + slopes[number - 1]
               coors[number - 1] = coor
               lengths[number - 1] = length 
            continue
        else:
            slopes.append(slope)
            coors.append(line[0])
            lengths.append(length)
    return coors

def contour_fit(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    useable_contours = list()
    
    # 未必要用contour解决可以尝试更好得方法
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > 100:
            new_shape = list(cnt.shape)
            del new_shape[1]
            cnt = np.reshape(cnt, newshape=new_shape)
            useable_contours.append(cnt)
            
    # 这里可能不止两个，这里只是测试使用得方案
    y = np.arange(0, 300, 1)  #  [0-300]
    x_prime = list()  # 列表
    for cnt in useable_contours:  # 可用轮廓
        x_coord = cnt[:, 0]  # 轮廓的x数值
        y_coord = cnt[:, 1]  #      y
        param = np.polyfit(y_coord, x_coord, 1)  # 已有轮廓拟合直线
        x_prime.append(np.polyval(param, y))  # 用y带入已知直线得到x_prime,两个轮廓
        
    x_prime = np.array(x_prime)
    x_average = np.average(x_prime, axis=0)  # 所有轮廓x平均
    mid_param = np.polyfit(x_average, y, 1)
    start_point = [0, int(np.polyval(mid_param, 0))]
    end_point = [299, int(np.polyval(mid_param, 299))]
    
    return start_point, end_point

while (cap.isOpened()):
    ret, img = cap.read()
    img_copy = cv2.resize(img, (height, width))
    opening = Image_process(img_copy)
    lines = cv2.HoughLinesP(opening, 1, np.pi/180, 100, minLineLength, maxLineGap)
    start_point, end_point = contour_fit(opening)
    cv2.line(img_copy, start_point, end_point, color=[255, 0, 0], thickness=2)
    Init_vary()
    coors = reduce_duplication(lines)
    for coor in coors:
         x1, y1, x2, y2 = coor
         cv2.line(img_copy,(x1, y1),(x2, y2),(0,255,0),2)
    
    y = np.arange(0, 300, 1)  #  [0-300]
    x_prime = list()  # 列表
    if len(coors) == 2:
        for coor in coors:
            x1, y1, x2, y2 = coor
            x_prime.append(x1 + (y - y1)*(x2 - x1)/(y2 - y1))
        x_prime = np.array(x_prime)
        x_average = np.average(x_prime, axis=0)  # 所有轮廓x平均
        mid_param = np.polyfit(x_average, y, 1)
        start_point = [0, int(np.polyval(mid_param, 0))]
        end_point = [299, int(np.polyval(mid_param, 299))]
        # cv2.line(img_copy, start_point, end_point, color=[255, 0, 0], thickness=2)
    cv2.imshow('opening', opening)
    cv2.imshow('img', img_copy)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()