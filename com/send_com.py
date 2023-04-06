import serial
import struct
import time
import numpy as np
ser = serial.Serial('COM12', 115200,
                         parity='N',
                         stopbits=1)

for i in np.arange(1.0, 10.0, 0.1):
    B =struct.pack('f', i)
    if len(B) ==4:
        print(B)