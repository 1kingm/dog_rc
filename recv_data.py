import serial
from time import sleep

def recv(serial):
    while True:
        data = serial.read_all().decode()  # str
        if data == '':
            continue
        else:
            break
        sleep(0.02)
    return data

if __name__ == '__main__':
    serial = serial.Serial('COM11', 115200, timeout=0.5)
    if serial.isOpen():
        print("serial open success")
    else:
        print("serial open failed")
    while True:
        data = recv(serial)
        print(data)  # str
