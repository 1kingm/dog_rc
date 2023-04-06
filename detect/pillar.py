# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import serial
import struct
import time
import sys
import USARTDefine as ud

global i
i = 2

threshold_dic = {'lower_seesaw' : np.array([7, 122, 48]), 'upper_seesaw' : np.array([33, 251, 145]),
                 'lower_handrail' : np.array([7, 122, 48]), 'upper_handrail' : np.array([33, 251, 145]),
                 'lower_floor' : np.array([0, 76, 69]), 'upper_floor' : np.array([78, 255, 226]),
                 'lower_dwb' : np.array([7, 122, 48]), 'upper_dwb' : np.array([33, 251, 145]),
                 'lower_black' : np.array([7, 122, 48]), 'upper_black' : np.array([33, 251, 145]),
                 'lower_red' : np.array([7, 122, 48]), 'upper_red' : np.array([33, 251, 145]),
                }

# 初�?�化串口
mpucom = serial.Serial(port='/dev/ttyUSB0',  # 串口，�?�使�??/dev/ttyTHS1作为移�?�代�??
                       baudrate=115200,      # 波特�??
                       bytesize=8,           # 数据�??
                       stopbits=1,           # 停�??�??
                       timeout=0.1,          # 间隔
                       parity='N')           # 校验�??
if (mpucom.is_open):
    print("串口开成功\r\n")
else:
    print("串口开失败\r\n")
    exit()

# 检测相机是否打开,并获取图�??
cam = cv2.VideoCapture('/dev/video0')
if cam.isOpened() == 0:
    print("Can't open the camera!")
else:
    print("Camera is opening!")
    width = 400
    height = 400

# 头帧
def data_frame_init():
    data_frame = [ord('R'), ord('C')]
    for i in range(len(data_frame), ud.CHECKSUM_BYTE + 1):
        data_frame += [0]
    data_frame[ud.MOVE_ENABLE_BYTE] = 1
    return data_frame

# 更改xy坐标的格�??
def change_xy(x, y, data_frame):
    data_frame[ud.X_BYTE:ud.X_BYTE + 2] = [x % 256, x // 256]
    data_frame[ud.Y_BYTE:ud.Y_BYTE + 2] = [y % 256, y // 256]
    # data_frame[0, ud.CHECKSUM_BYTE:ud.CHECKSUM_BYTE+1] = np.sum(data_frame[0, :], axis=0) % 256
    return data_frame

# 校验�??
def cal_checksum(data_frame):
    checksum = sum(data_frame[:-1])
    checksum %= 128
    return checksum

# �??定义编码格式
def my_encode(list_obj):
    result = bytearray()
    for i in range(len(list_obj)):
        result += bytearray(chr(list_obj[i]).encode('utf-16'))[2:3][::-1]
    return result

def send_my_code(x, y, data_frame):
    x_bias = x
    y_bias = y
    data_frame = change_xy(x_bias, y_bias, data_frame=data_frame)
    data_frame[-1] = cal_checksum(data_frame)
    data_frame = my_encode(data_frame)
    mpucom.write(data_frame)
    print(data_frame)

# 检测现在的障�?�物�??�??�??
# def the_boundary_right_now():
#     boundary_list = ['seesaw', 'red1', 'handrail', 'red2', 'floor', 'red3', 'dwb', 'red4']
#     buff = mpucom.inWaiting()
#     data = b''
#     if (len(buff) > 0):
#         data = mpucom.read(buff)
#         time.sleep(0.02)
#     # isStop = ord(data[5])  # 假�?��??五位给是否停止的指令
#     return isStop, i

# 检测跷跷板
def seesaw_detector():
    pass

# 检测楼�??
def floor_detector():
    # isStop, i = the_boundary_right_now()
    while (i == 2):
        ret, frame = cam.read()

        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 5*5高斯滤波
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # �??换色�??
        # cv2.flip(frame, -1, frame)
        mask = cv2.inRange(hsv, threshold_dic['lower_floor'], threshold_dic['upper_floor'])
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        # cv2.flip(mask, -1, mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # 查找检测物体的�??�??

        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)  # 取最大的�??�??
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w > 0 and h < 10000:  # 如果半径大于50�??像素且小�??70�??像素，具体数值待�??
                # 首先传递停下的指令
                # if (isStop == 0):
                #     data_frame = data_frame_init()
                #     send_my_code(0, 0, data_frame)
                #     print('Stop')
                #     isStop = 1

                # 将�?�测到的�??�色标�?�出�??

                x_bias = int(x + w / 2)
                y_bias = int(y + h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1, cv2.LINE_AA)

                # 处理数据并发�??
                data_frame = data_frame_init()
                send_my_code(x_bias, y_bias, data_frame)
                print('sending')

                time.sleep(0.02)
        else:
            data_frame = data_frame_init()
            send_my_code(0, 0, data_frame)
            print('No conrours found')

        # cv2.imshow('floor_mask', mask)
        # cv2.imshow('floor_frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    floor_detector()
    cam.release()
    cv2.destroyAllWindows()

