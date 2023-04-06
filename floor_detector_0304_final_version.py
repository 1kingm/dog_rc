# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import serial
import struct
import time
import sys

# ��ʼ������
global i
i = 2


head_frame = 'RC'



# ��ʼ����ֵ
threshold_dic = {'lower_seesaw' : np.array([7, 122, 48]), 'upper_seesaw' : np.array([33, 251, 145]),
                 'lower_handrail' : np.array([7, 122, 48]), 'upper_handrail' : np.array([33, 251, 145]),
                 'lower_floor' : np.array([7, 122, 48]), 'upper_floor' : np.array([33, 251, 145]),
                 'lower_dwb' : np.array([7, 122, 48]), 'upper_dwb' : np.array([33, 251, 145]),
                 'lower_black' : np.array([7, 122, 48]), 'upper_black' : np.array([33, 251, 145]),
                 'lower_red' : np.array([7, 122, 48]), 'upper_red' : np.array([33, 251, 145]),
                }

# ��ʼ������
mpucom = serial.Serial(port='/dev/ttyUSB1',  # ���ڣ���ʹ��/dev/ttyTHS1��Ϊ��ֲ����
                       baudrate=115200,      # ������
                       bytesize=8,           # ����λ
                       stopbits=1,           # ֹͣλ
                       timeout=0.1,          # ���
                       parity='N')           # У��λ
if (mpucom.is_open):
    print("���ڿ����ɹ�\r\n")
else:
    print("���ڿ���ʧ��\r\n")
    exit()

# �������Ƿ��,����ȡͼ��
cam = cv2.VideoCapture(0)
if cam.isOpened() == 0:
    print("Can't open the camera!")
else:
    print("Camera is opening!")
    width = 400
    height = 400

# ������ڵ��ϰ������ĸ�
def the_boundary_right_now():
    boundary_list = ['seesaw', 'red1', 'handrail', 'red2', 'floor', 'red3', 'dwb', 'red4']
    i = mpucom.read(1)
    if (i >= 0):
        isStop = 0
    else:
        isStop = 1
    return isStop, i

# ��ʮ���Ʊ�ɶ����Ʋ���ȫΪʮ��λ
def decimal2bin(number):
    s = []
    binString = ''
    while number > 0:
        rem = number % 2
        s.append(rem)
        number = number // 2
    while len(s) > 0:
        binString = binString + str(s.pop())
    binString = binString.rjust(16, "0")
    return binString


def send(com, data):
    



# ������ΰ�
def seesaw_detector():
    isStop, i = the_boundary_right_now()
    while (i == 2):
        ret, frame = cam.read()

        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 5*5��˹�˲�
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # ת��ɫ��
        cv2.flip(frame, -1, frame)
        mask = cv2.inRange(hsv, threshold_dic['lower_seesaw'], threshold_dic['upper_seesaw'])
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        cv2.flip(mask, -1, mask)
        cv2.imshow('floor_mask', mask)
        cv2.imshow('floor_frame', frame)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # ���Ҽ�����������

        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)  # ȡ��������
            (color_x, color_y), color_radius = cv2.minEnclosingCircle(cnt)  # Ѱ�Ұ�����������СԲ������Բ������Ͱ뾶
            if color_radius > 0 and color_radius < 100000:  # ����뾶����50��������С��70�����أ�������ֵ����
                # ���ȴ���ͣ�µ�ָ��
                if (isStop == 0):
                    print('S')
                    mpucom.write('S'.encode())
                    isStop = 1

                # ����⵽����ɫ��ǳ���
                cv2.circle(frame, (int(color_x), int(color_y)), int(color_radius), (255, 0, 255), 2)
                x_bias = int(color_x)
                y_bias = int(color_x)

                x_bias_send = decimal2bin(x_bias)
                y_bias_send = decimal2bin(y_bias)

                mpucom.write('X'.encode('utf-8') + x_bias_send.encode() + 'E'.encode('utf-8'))
                mpucom.write('Y'.encode('utf-8') + y_bias_send.encode() + 'E'.encode('utf-8'))
                print('X'.encode() + x_bias_send.encode() + 'E'.encode())
                print('Y'.encode() + y_bias_send.encode() + 'E'.encode())
                time.sleep(0.02)
        else:
            print('F')
            mpucom.write('F'.encode())
            mpucom.write('F'.encode('utf-8'))

        if cv2.waitKey(1) == ord('q'):
            break

# ���¥��
def floor_detector():
    isStop, i = the_boundary_right_now()
    while (i == 2):
        ret, frame = cam.read()

        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 5*5��˹�˲�  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # ת��ɫ��  
        cv2.flip(frame, -1, frame)  # ͼ��ת
        mask = cv2.inRange(hsv, threshold_dic['lower_floor'], threshold_dic['upper_floor'])
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        cv2.flip(mask, -1, mask)  # ͼ���ٴη�ת����ԭ
        cv2.imshow('floor_mask', mask)
        cv2.imshow('floor_frame', frame)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # ���Ҽ�����������

        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)  # ȡ��������
            (color_x, color_y), color_radius = cv2.minEnclosingCircle(cnt)  # Ѱ�Ұ�����������СԲ������Բ������Ͱ뾶
            if color_radius > 0 and color_radius < 100000:  # ����뾶����50��������С��70�����أ�������ֵ����
                # ���ȴ���ͣ�µ�ָ��
                if (isStop == 0):
                    print('S')
                    mpucom.write('S'.encode())
                    isStop = 1

                # ����⵽����ɫ��ǳ���
                cv2.circle(frame, (int(color_x), int(color_y)), int(color_radius), (255, 0, 255), 2)
                x_bias = int(color_x)
                y_bias = int(color_x)

                x_bias_send = decimal2bin(x_bias)
                y_bias_send = decimal2bin(y_bias)

                mpucom.write('X'.encode('utf-8') + x_bias_send.encode() + 'E'.encode('utf-8'))
                mpucom.write('Y'.encode('utf-8') + y_bias_send.encode() + 'E'.encode('utf-8'))
                print('X'.encode() + x_bias_send.encode() + 'E'.encode())
                print('Y'.encode() + y_bias_send.encode() + 'E'.encode())
                time.sleep(0.02)
        else:
            print('F')
            mpucom.write('F'.encode())
            mpucom.write('F'.encode('utf-8'))

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    floor_detector()
    cam.release()
    cv2.destroyAllWindows()

