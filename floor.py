import cv2
import numpy as np
import USARTDefine as ud
import serial
from imutils import perspective
import time
import sys
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 膨胀腐蚀操作卷积核大小
threshold_dic = {'lower_seesaw' : np.array([7, 122, 48]), 'upper_seesaw' : np.array([33, 251, 145]),
                 'lower_handrail' : np.array([7, 122, 48]), 'upper_handrail' : np.array([33, 251, 145]),
                 'lower_floor' : np.array([0, 45, 0]), 'upper_floor' : np.array([35, 255, 255]),
                 'lower_dwb' : np.array([7, 122, 48]), 'upper_dwb' : np.array([33, 251, 145]),
                 'lower_black' : np.array([7, 122, 48]), 'upper_black' : np.array([33, 251, 145]),
                 'lower_red' : np.array([0, 62, 0]), 'upper_red' : np.array([255, 255, 255]),
                }

cap = cv2.VideoCapture(1)
# mpucom = serial.Serial(port='COM3',  # 串口，请使用/dev/ttyUSB0作为移植代码
#                        baudrate=115200,      # 波特率
#                        bytesize=8,           # 数据位
#                        stopbits=1,           # 停止位
#                        timeout=0.1,          # 间隔
#                        parity='N')           # 校验位
def about_equal(x, y, percent):
    if (x >= y*(1-percent)) & (x <= y*(1+percent)):
        return 1
    else:
        return 0
# # 头帧
# def data_frame_init(isStop):
#     data_frame = [ord('R'), ord('C')]
#     for i in range(len(data_frame), ud.CHECKSUM_BYTE + 1):
#         data_frame += [0]
#     if (isStop == 0):  # 停止位
#         data_frame[ud.MOVE_ENABLE_BYTE] = 1
#     else:
#         data_frame[ud.MOVE_ENABLE_BYTE] = 0
#     return data_frame

# # 更改xy坐标的格式
# def change_xyk(x, y, k, data_frame):
#     data_frame[ud.X_BYTE:ud.X_BYTE + 2] = [x % 256, x // 256]
#     data_frame[ud.Y_BYTE:ud.Y_BYTE + 2] = [y % 256, y // 256]
#     data_frame[ud.FLOAT_DATA_BYTE:ud.FLOAT_DATA_BYTE + 2] = [k % 256, k // 256]
#     # data_frame[0, ud.CHECKSUM_BYTE:ud.CHECKSUM_BYTE+1] = np.sum(data_frame[0, :], axis=0) % 256
#     return data_frame

# # 校验和
# def cal_checksum(data_frame):
#     checksum = sum(data_frame[:-1])
#     checksum %= 128
#     return checksum

# # 自定义编码格式
# def my_encode(list_obj):
#     result = bytearray()
#     for i in range(len(list_obj)):
#         result += bytearray(chr(list_obj[i]).encode('utf-16'))[2:3][::-1]
#     return result

# def send_my_code(x, y, k, data_frame):
    x_bias = x
    y_bias = y
    k = k
    data_frame = change_xyk(x_bias, y_bias, k, data_frame=data_frame)
    data_frame[-1] = cal_checksum(data_frame)
    data_frame = my_encode(data_frame)
    mpucom.write(data_frame)
    print(data_frame)
 

while True:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, threshold_dic['lower_floor'], threshold_dic['upper_floor'])
    mask_median = cv2.medianBlur(mask, 3)
    mask_open = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)
    k, jump = 0, 0
    
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
        
        approxs = cv2.approxPolyDP(contours[max_id], 10, True)  # 多边形逼近轮廓, 精度10像素
        cv2.polylines(frame, [approxs], True, (0, 0, 255), 3)
        # 这里放一个获取下面两坐标
        box = perspective.order_points(max_box)
        bot1 = box[2]
        bot2 = box[3]
        print(bot1[1])
        print(bot2[1])
        if about_equal(bot1[1], bot2[1], 0.1):
            # 前进
            if bot1[1] >= frame.shape[0] * 0.80:
                jump = 1
        else:# 旋转
            k = (bot1[1] - bot2[1])/(bot1[0] - bot2[0])
    except:
        print("未检测到轮廓")
    # cv2.imshow('frame', frame)
    # wait = cv2.waitKey(1)
    # if wait == 27:
    #     break
cv2.destroyAllWindows()
cap.release()