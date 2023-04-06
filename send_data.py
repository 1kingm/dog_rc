import numpy as np
import USARTDefine as ud
import cv2
import serial
import numpy as np
import time
import struct

frame_head = str('RC')
x_byte = [0, 0]
y_byte = [0, 0]
task_byte = 0
float_data_byte = [0, 0, 0, 0]
move_enable_byte = 0

COM_name = '/dev/ttyUSB0'  # 串口名称 win版本为com+n，linux为/ttyUSB+n

mpucom = serial.Serial(COM_name, 115200,
                         parity='N',
                         stopbits=1)  # 串口操作

def list2str(list_obj):
    string = str()
    for i in range(len(list_obj)):
        string += chr(list_obj[i])
    return string

def cal_checksum(data_frame):
    checksum = sum(data_frame)
    checksum %= 128
    return checksum

# 更改xy坐标的格式
def change_xyk(x, y, k, data_frame):
    data_frame[ud.X_BYTE:ud.X_BYTE + 2] = [x % 256, x // 256]
    data_frame[ud.Y_BYTE:ud.Y_BYTE + 2] = [y % 256, y // 256]
    data_frame[ud.FLOAT_DATA_BYTE:ud.FLOAT_DATA_BYTE + 4] = struct('f', k)
    # data_frame[0, ud.CHECKSUM_BYTE:ud.CHECKSUM_BYTE+1] = np.sum(data_frame[0, :], axis=0) % 256
    return data_frame

def data_frame_init(isStop):
    data_frame = [ord('R'), ord('C')]
    for i in range(len(data_frame), ud.CHECKSUM_BYTE + 1):
        data_frame += [0]
    if (isStop == 0):  # 停止位
        data_frame[ud.MOVE_ENABLE_BYTE] = 1
    else:
        data_frame[ud.MOVE_ENABLE_BYTE] = 0
    return data_frame

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