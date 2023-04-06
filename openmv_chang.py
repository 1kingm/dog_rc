# Untitled - By: js'slover - 周四 3月 30 2023

# 导入需要的库
import seekfree, pyb
import sensor, image, time, math
import os, tf
from machine import UART  # 串口
import pyb


# 初始化摄像头
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # GRAYSCALE RGB565 BAYER JPEG
sensor.set_framesize(sensor.QQVGA)  # 像素大小160*120
sensor.set_brightness(300) # -3 to 3 亮度
sensor.skip_frames(time = 20) # 跳过初始帧
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(True,(0,0,0)) # 调节白平衡

# 初始化串口
uart = UART(2, baudrate=115200)     # 初始化串口 波特率设置为115200 TX是B12 RX是B13

# 基本变量初始化
clock = time.clock() # 计算帧率
current_task = 0
position = [] # 坐标点的xy
position_num = 0 # 坐标点的个数
count_same_num = 0 # 四次计数相同发送
center_x = 0
center_y = 0
A, B, C = 0, 0, 0
net_path = "model.tflite"                                  # 定义模型的路径
labels = [line.rstrip() for line in open("/sd/labels.txt")]   # 加载标签
net = tf.load(net_path, load_to_fb=True)                                  # 加载模型


def detect_A4(img): # 检测A4纸上的坐标
    global position
    global count_same_num
    global current_task
    # global position_num
    posx,posy=0,0
    for r in img.find_rects(threshold = 20000):
        img.draw_rectangle(r.rect(), color = (255, 0, 0))   # 绘制红色矩形框
        img_x=(int)(r.rect()[0]+r.rect()[2]/2)              # 图像中心的x值
        img_y=(int)(r.rect()[1]+r.rect()[3]/2)              # 图像中心的y值
        img.draw_circle(img_x, img_y, 5, color = (0, 255, 0)) # 给矩形中心绘制一个小圆 便于观察矩形中心是否识别正确
        for c in img.find_circles(roi = r.rect(), threshold = 1000, x_margin = 5, y_margin = 5, r_margin = 5,r_min = 3, r_max = 5, r_step = 1):
            # 绘制圆形外框
            img.draw_circle(c.x(), c.y(), c.r(), color = (255, 0, 0))
            # 计算实际X坐标点
            posx = (float(c.x()) - float(r.x())) / float(r.w()) * 35 + 0.5
            # 计算实际Y坐标点
            posy = 25 - ((float(c.y()) - float(r.y())) / float(r.h()) * 25) + 0.5
            position_num_past = position_num
            position = Deduplication(int(posx), int(posy), position)
            if (position_num == position_num_past) & (position_num != 0):
                count_same_num = count_same_num + 1
            if count_same_num == 4:
                send_position(position)
                current_task = 1

def Deduplication(x, y, position_last): # 当前检测坐标加入坐标点集
    global position_num
    if (x>=35 & y <=1) | (x>=35 & y>=25) | (x<=1 & y<=1) | (x<=1 & y >=25):  # 边缘被异常检测为点
        return position_last
    position_new = (x, y)
    if position_num == 0: # 点集中没有任何点
        position_num = position_num + 1
        position_last.append(position_num)
        position_last.extend(position_new)
        return position_last
    else:
        if position_new in position_last: # 当前点已存在
            return position_last
        else:
            position_num = position_num + 1
            position_last.append(position_num)
            position_last.extend(position_new) # 能加进来也是不容易
            return position_last

def send_position(position): # 发送坐标数据
    position.insert(0, 122)
    position.append(98)
    uart.write(bytearray(position))

def fine_tuning(img): # 微调
    global current_task
    uart_num = uart.any()
    time.sleep(1000)
    uart_str = ''
    if uart_num:
        uart_str = uart.read(uart_num)
    if uart_str != 'S':
        pass
    else:
        roi = find_roi(img)
        if roi == 0:
            pass
        else:
            img_x=(int)(roi[0]+roi[2]/2)              # 图像中心的x值
            img_y=(int)(roi[1]+roi[3]/2)              # 图像中心的y值
            offset_dot = length_point(img_x, img_y)
            uart.write(bytearray(offset_dot))
            if offset_dot == [0, 0]:
                current_task = 2
        
def find_roi(img):
    rois = []
    roi_area = []
    number = 0
    roi = []
    for r in img.find_rects(threshold = 10000,connectivity=2):             # 在图像中搜索矩形
         if (r.rect()[2] <= r.rect()[3]*1.05) & (r.rect()[2] >= r.rect()[3]*0.95):
            img.draw_rectangle(r.rect(), color = (255, 0, 0))   # 绘制矩形外框，便于在IDE上查看识别到的矩形位置
            rois.append(r.rect())
            number = number + 1
    if len(rois) != 0 & number != 1:
        for roi in rois:
            roi_area.append(roi[2]*roi[3])
        roi = rois[roi_area.index(max(roi_area))]                        # 拷贝矩形框内的图像
    elif len(rois) != 0 & number != 1:
        roi = rois
    return roi

def length_point(x, y):
    if math.sqrt((x-center_x)*(x-center_x)+(y-center_y)*(y-center_y))<10:
        return 0, 0
    else:
        return x-center_x,y-center_y

def class_img(img):
    global A, B, C, current_task
    uart_num = uart.any()
    time.sleep(1000)
    uart_str = ''
    _class = ''
    if uart_num:
        uart_str = uart.read(uart_num)
    if uart_str != 'C':
        pass
    else:
        img_roi = img(find_roi(img))
        for obj in tf.classify(net , img_roi, min_scale=1.0, scale_mul=0.5, x_overlap=0.0, y_overlap=0.0):
                print("**********\nTop 1 Detections at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
                sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
                # 打印准确率最高的结果
                print("%s = %f" % (sorted_list[0][0], sorted_list[0][1]))
                if sorted_list[0][1] <= 0.6:
                    _class = 'None'
                else:
                    if _class == 'cat' or _class == 'dog' or _class == 'cow' or _class == 'pig' or _class == 'casttle':
                        A = A + 1
                        if A == 5:
                            class_send = [67, 1, 83]
                            uart.write(bytearray(class_send))
                            A, B, C = 0, 0, 0
                            current_task = 1
                    if _class == 'airplane' or _class == 'boat' or _class == 'car' or _class == 'train' or _class == 'casttle':
                        B = B + 1
                        if B == 5:
                            class_send = [67, 2, 83]
                            uart.write(bytearray(class_send))
                            A, B, C = 0, 0, 0
                            current_task = 1
                    if _class == 'apple' or _class == 'banana' or _class == 'durian' or _class == 'grape' or _class == 'orange':
                        C = C + 1
                        if C == 5:
                            class_send = [67, 3, 83]
                            uart.write(bytearray(class_send))
                            A, B, C = 0, 0, 0
                            current_task = 1


judge_task = {
    0:detect_A4,
    1:fine_tuning,
    2:class_img
}

while True:
    clock.tick()
    img = sensor.snapshot()
    judge_task[current_task](img)
