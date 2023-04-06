# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import serial
import struct
import time
import sys
import USARTDefine as ud

# 初始化变量
global i
i = 2

# 初始化阈值
threshold_dic = {'lower_seesaw' : np.array([7, 122, 48]), 'upper_seesaw' : np.array([33, 251, 145]),
                 'lower_handrail' : np.array([7, 122, 48]), 'upper_handrail' : np.array([33, 251, 145]),
                 'lower_floor' : np.array([14, 71, 56]), 'upper_floor' : np.array([54, 121, 188]),
                 'lower_dwb' : np.array([7, 122, 48]), 'upper_dwb' : np.array([33, 251, 145]),
                 'lower_black' : np.array([7, 122, 48]), 'upper_black' : np.array([33, 251, 145]),
                 'lower_red' : np.array([0, 62, 0]), 'upper_red' : np.array([255, 255, 255]),
                }

# 初始化串口
mpucom = serial.Serial(port='COM3',  # 串口，请使用/dev/ttyUSB0作为移植代码
                       baudrate=115200,      # 波特率
                       bytesize=8,           # 数据位
                       stopbits=1,           # 停止位
                       timeout=0.1,          # 间隔
                       parity='N')           # 校验位
if (mpucom.is_open):
    print("串口开启成功\r\n")
else:
    print("串口开启失败\r\n")
    exit()

# 检测相机是否打开,并获取图像
cam = cv2.VideoCapture(1)
if cam.isOpened() == 0:
    print("Can't open the camera!")
else:
    print("Camera is opening!")
    width = 400
    height = 400

# 头帧
def data_frame_init(isStop):
    data_frame = [ord('R'), ord('C')]
    for i in range(len(data_frame), ud.CHECKSUM_BYTE + 1):
        data_frame += [0]
    if (isStop == 0):  # 停止位
        data_frame[ud.MOVE_ENABLE_BYTE] = 1
    else:
        data_frame[ud.MOVE_ENABLE_BYTE] = 0
    return data_frame

# 更改xy坐标的格式
def change_xyk(x, y, k, data_frame):
    data_frame[ud.X_BYTE:ud.X_BYTE + 2] = [x % 256, x // 256]
    data_frame[ud.Y_BYTE:ud.Y_BYTE + 2] = [y % 256, y // 256]
    data_frame[ud.FLOAT_DATA_BYTE:ud.FLOAT_DATA_BYTE + 2] = [k % 256, k // 256]
    # data_frame[0, ud.CHECKSUM_BYTE:ud.CHECKSUM_BYTE+1] = np.sum(data_frame[0, :], axis=0) % 256
    return data_frame

# 校验和
def cal_checksum(data_frame):
    checksum = sum(data_frame[:-1])
    checksum %= 128
    return checksum

# 自定义编码格式
def my_encode(list_obj):
    result = bytearray()
    for i in range(len(list_obj)):
        result += bytearray(chr(list_obj[i]).encode('utf-16'))[2:3][::-1]
    return result

def send_my_code(x, y, k, data_frame):
    x_bias = x
    y_bias = y
    k = k
    data_frame = change_xyk(x_bias, y_bias, k, data_frame=data_frame)
    data_frame[-1] = cal_checksum(data_frame)
    data_frame = my_encode(data_frame)
    mpucom.write(data_frame)
    print(data_frame)

# 检测现在的障碍物是哪个
# def the_boundary_right_now():
#     boundary_list = ['seesaw', 'red1', 'handrail', 'red2', 'floor', 'red3', 'dwb', 'red4']
#     buff = mpucom.inWaiting()
#     data = b''
#     if (len(buff) > 0):
#         data = mpucom.read(buff)
#         time.sleep(0.02)
#     # isStop = ord(data[5])  # 假设第五位给是否停止的指令
#     return isStop, i

def red_detector():
    x_bias, y_bias = 0, 0
    while (i == 2):
        left_most_list = []
        right_most_list = []
        top_most_list = []
        death_zone = 10

        ret, img = cam.read()

        width = 500
        height = 600

        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        half_height = height / 2
        half_width = width / 2
        right_line_most_left_point = width
        left_line_most_right_point = 0
        top_line_most_top_point = height
        img_rgb = cv2.GaussianBlur(img, (5, 5), 0)  # 5*5 gaussian blur
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)  # BGR to HSV
        mask_red_1 = cv2.inRange(img_hsv, threshold_dic['lower_red'], threshold_dic['upper_red'])
        mask_red_2 = cv2.GaussianBlur(mask_red_1, (3, 3), 0)

        cv2.imshow('floor_mask', mask_red_2)
        cv2.imshow('floor_frame', img)

        contours, hierarchy = cv2.findContours(mask_red_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # print("the type of contours is " + str(type(contours)))  # tuple
        # print(contours)
        # print(contours[i][j])  # first [i] means the (i+1)th contour(the bigger the former), second [j] means the (i+1)th pixel(upper left corner point is the first)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(
                cnt)  # xy are the coordinate of upper left corner point，w stands for width，h stand for height
            if (w > 100 and h > 100):
                left_most = tuple(cnt[cnt[:, :, 0].argmin()][0])  # warning! The type of left_most if numpy.intc != int
                # left_most_the_same = cnt[:, :, 0].min()
                left_most_list.append(left_most)

                if (left_most[0] > half_width and left_most[0] < right_line_most_left_point):
                    right_line_most_left_point = left_most[0]

                right_most = tuple(cnt[cnt[:, :, 0].argmax()][0])
                right_most_list.append(right_most)

                if (right_most[0] < half_width and right_most[0] > left_line_most_right_point):
                    left_line_most_right_point = right_most[0]

                top_most = tuple(cnt[cnt[:, :, 1].argmin()][0])
                top_most_list.append(top_most)

                if (top_most[1] < top_line_most_top_point):
                    top_line_most_top_point = top_most[0]

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(img, left_most, 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(img, right_most, 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(img, top_most, 5, (0, 255, 0), -1, cv2.LINE_AA)

        if (len(left_most_list) >= 2 and len(right_most_list) >= 2):
            mid_point_1_x = int((left_most_list[0][0] + right_most_list[1][0]) / 2)
            mid_point_1_y = int((left_most_list[0][1] + right_most_list[1][1]) / 2)
            mid_point_1 = (mid_point_1_x, mid_point_1_y)
            mid_point_2_x = int((left_most_list[1][0] + right_most_list[0][0]) / 2)
            mid_point_2_y = int((left_most_list[1][1] + right_most_list[0][1]) / 2)
            mid_point_2 = (mid_point_2_x, mid_point_2_y)
            cv2.line(img, mid_point_1, mid_point_2, (0, 255, 0), 3, cv2.LINE_AA)
            if (mid_point_2[0] - mid_point_1[0] != 0):
                k = (mid_point_2[1] - mid_point_1[1]) / (
                            mid_point_2[0] - mid_point_1[0])  # if k is positive you need turn left
                k = int(k)
                if (abs(k) > death_zone):
                    isStop = 0
                    data_frame = data_frame_init(isStop)
                    send_my_code(x_bias, y_bias, k, data_frame)
                else:
                    k = 0
                    isStop = 0
                    data_frame = data_frame_init(isStop)
                    send_my_code(x_bias, y_bias, k, data_frame)
            else:
                k = 0
                isStop = 0
                data_frame = data_frame_init(isStop)
                send_my_code(x_bias, y_bias, k, data_frame)

        # width_mid_line = (left_line_most_right_point + right_line_most_left_point) / 2
        # x_bias = half_height - width_mid_line  # bigger than 0 means we need turn right
        # print(x_bias)

        if (len(top_most_list) == 3):
            if (top_most_list[2][1] > half_height / 10):
                k = 0
                isStop = 1
                data_frame = data_frame_init(isStop)
                send_my_code(x_bias, y_bias, k, data_frame)

        if (len(top_most_list) == 1):
            if (top_most_list[0][1] > half_height / 10):
                k = 0
                isStop = 1
                data_frame = data_frame_init(isStop)
                send_my_code(x_bias, y_bias, k, data_frame)

        cv2.imshow('contours', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 检测跷跷板
def seesaw_detector():
    pass

# 检测楼梯
def floor_detector():
    pass
    # isStop, i = the_boundary_right_now()
    while (i == 2):
        ret, frame = cam.read()
    
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 5*5高斯滤波
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换色域
        # cv2.flip(frame, -1, frame
        # )
        mask = cv2.inRange(hsv, threshold_dic['lower_floor'], threshold_dic['upper_floor'])
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        cv2.imshow('floor_mask', mask)
    
        cv2.waitKey(1)
    
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # 查找检测物体的轮廓
    
        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)  # 取最大的轮廓
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w > 0 and h > 0:  # 如果半径大于50个像素且小于70个像素，具体数值待定
                x_bias = int(x + w / 2)
                y_bias = int(y + h / 2)
    
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1, cv2.LINE_AA)
    
                # 处理数据并发送
                data_frame = data_frame_init()
                send_my_code(x_bias, y_bias, data_frame)
                print('sending')
    
                time.sleep(0.02)
        else:
            data_frame = data_frame_init()
            send_my_code(0, 0, data_frame)
            print('No contours found')
    
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    red_detector()
    seesaw_detector()
    cam.release()
    cv2.destroyAllWindows()

