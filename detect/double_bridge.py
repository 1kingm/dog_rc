import cv2
import numpy as np
import math
import task

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 膨胀腐蚀操作卷积核大小
hsv_low = np.array([15, 42, 172])
hsv_high = np.array([45, 116, 255])
_jump1 = False
_jump2 = False
goal = 2
counter, counter2 = 0, 0
counter_bool, counter2_bool = False, False


def length_line(dot1, dot2):
    length = math.sqrt((dot1[0] - dot2[0]) * (dot1[0] - dot2[0]) + (dot1[1] - dot2[1]) * (dot1[1] - dot2[1]))
    return length


def nei_abf(dot1, dot2):  # 计算斜率和截距
    Nei = (dot1[0] - dot2[0]) / (dot1[1] - dot2[1])  # 斜率,
    abf = dot1[0] - Nei * dot1[1]  # 截距
    return [Nei, abf]  # x/y 默认斜率算法


def init_adjust(nei_med, x):  # nei_med 是斜率
    global counter2, counter2_bool
    if x < 330:
        # print("axs")
        counter2_bool = False
        return task.ROTAT_LEFT, nei_med, x
    elif x > 400:
        # print("xaxs")
        counter2_bool = False
        return task.ROTAT_RIGHT, nei_med, x
    else:
        # print('success')
        if counter2_bool:
            counter2 = counter2 + 1
        else:
            counter2 = 0
        counter2_bool = True
        return -1, -1, -1


def rect_dot(frame, area1, area2, number):
    rect1 = cv2.minAreaRect(area1)
    rect2 = cv2.minAreaRect(area2)

    conflag1, conflag2 = 0.01, 0.01
    img_ele1, img_ele2 = conflag1 * cv2.arcLength(area1, True), conflag2 * cv2.arcLength(area2, True)
    approxCourve1 = cv2.approxPolyDP(area1, img_ele1, closed=True)
    approxCourve2 = cv2.approxPolyDP(area2, img_ele2, closed=True)

    while True:  # 检测多边形为小于等于四条边的
        if len(approxCourve1) <= 4:
            break
        else:
            conflag1 = conflag1 * 1.5
            img_ele1 = conflag1 * cv2.arcLength(area1, True)
            approxCourve1 = cv2.approxPolyDP(area1, img_ele1, closed=True)
    while True:
        if len(approxCourve2) <= 4:
            break
        else:
            conflag2 = conflag2 * 1.5
            img_ele2 = conflag2 * cv2.arcLength(area1, True)
            approxCourve2 = cv2.approxPolyDP(area2, img_ele2, closed=True)
    frame = cv2.polylines(frame, [approxCourve1], True, color=(0, 255, 255), thickness=3)
    frame = cv2.polylines(frame, [approxCourve2], True, color=(255, 255, 255), thickness=3)
    cv2.imshow('cnts', frame)
    cv2.waitKey(1)
    approxCourve1 = np.array(approxCourve1)
    approxCourve1 = np.squeeze(approxCourve1, axis=1)
    approxCourve2 = np.array(approxCourve2)
    approxCourve2 = np.squeeze(approxCourve2, axis=1)

    # cv2.line(frame, approxCourve2[2], approxCourve2[1], (0, 255, 255), 2)
    dot_num1, dot_num2 = len(approxCourve1), len(approxCourve2)
    if rect1[0][0] > rect2[0][0]:  # 判断两个矩形位置，根据位置确定取的直线, rect1是右边的，rect2是左边的
        if dot_num1 == 2 | dot_num2 == 2:
            if number == 1:
                return 0
            elif number == 2:
                return [0, 0], [0, 0]
            elif number == 3:
                return approxCourve1[0][1] + approxCourve2[0][1]
        else:
            if number == 1:
                return length_line(approxCourve1[2], approxCourve1[dot_num1 - 1]) + \
                       length_line(approxCourve2[2], approxCourve2[1])
            elif number == 2:
                return nei_abf(approxCourve1[0], approxCourve1[1]), \
                       nei_abf(approxCourve2[0], approxCourve2[dot_num2 - 1])
            elif number == 3:
                return approxCourve1[0][1] + approxCourve2[0][1]
    else:
        if dot_num1 == 2 | dot_num2 == 2:
            if number == 1:
                return 0
            elif number == 2:
                return [0, 0], [0, 0]
            elif number == 3:
                return approxCourve1[0][1] + approxCourve2[0][1]
        else:
            if number == 1:
                return length_line(approxCourve1[2], approxCourve1[1]) + \
                       length_line(approxCourve2[2], approxCourve2[dot_num2 - 1])
            elif number == 2:
                # print(2)
                return nei_abf(approxCourve1[0], approxCourve1[dot_num1 - 1]), \
                       nei_abf(approxCourve2[0], approxCourve2[1])
            elif number == 3:
                return approxCourve1[0][1] + approxCourve2[0][1]


def double_bridge(frame):
    global _jump1, _jump2
    global goal, counter_bool, counter
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, hsv_low, hsv_high)

    mask_median = cv2.medianBlur(mask, 3)
    mask_open = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)
    cv2.imshow('mask', mask_open)
    cv2.waitKey(1)
    contours, _hierarchy = cv2.findContours(mask_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓查找

    cnts = []  # 剔除部分之后的全部轮廓
    for cnt in range(len(contours)):  # 剔除部分轮廓
        (x, y, w, h) = cv2.boundingRect(contours[cnt])
        if w > 10 and h > 30:
            cnts.append(contours[cnt])

    area = []  # 轮廓面积
    for cnt in cnts:
        area.append(cv2.contourArea(cnt))

    # area为所有的cnt的面积
    area_sort = area
    area_sort.sort(reverse=True)  # 根据面积序降的area，排序结果

    if len(cnts) < 2:
        if counter_bool:
            counter = 0
        else:
            counter = counter + 1
        counter_bool = False
        if counter == 40:
            return task.Jump_Standard, -1, -1
        else:
            return -1, -1, -1
    else:
        counter_bool = True  # 上一帧是cnts>2, 下一帧清空

    area1 = cnts[area.index(area_sort[0])]  # 最大的轮廓
    area2 = cnts[area.index(area_sort[1])]  # 第二大的轮廓

    [nei1, abf1], [nei2, abf2] = rect_dot(frame, area1=area1, area2=area2, number=2)
    if [nei1, abf1] == [0, 0] and [nei2, abf2] == [0, 0]:
        return -1, -1, -1
    # TODO 根据实际图像尺寸修改
    y_coors = np.arange(1, 480)
    x_coors = []

    for y in y_coors:
        x_coor = (y * nei1 + abf1 + y * nei2 + abf2) / 2
        x_coors.append(x_coor)
        cv2.circle(frame, (int(x_coor), int(y)), 2, (0, 0, 255), -1)

    nei_median, intercept = np.polyfit(y_coors, x_coors, 1)  # 拟合直线
    x_mid = nei_median * (frame.shape[1] // 2) + intercept
    if counter2 < 30:
        init_adjust(1 / nei_median, x_mid)  # 判断初始位置是否正确
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if goal == 1:
        if nei_median == 0:
            k_median = 10000
        else:
            k_median = 1 / nei_median

        if rect_dot(frame, area1, area2, 1) >= 40 and not _jump1:  # TODO根据最下的判断是否可以跳跃
            goal = 2
            _jump1 = True
            return task.Jump_Standard, -1, -1
        else:
            return task.VISION, k_median, x_mid
    elif goal == 2:
        if nei_median == 0:
            k_median = 10000
        else:
            k_median = 1 / nei_median

        if rect_dot(frame, area1, area2, 3) > 10000 and not _jump2:  # TODO 判断跳跃条件
            goal = 3
            _jump2 = True
            # print(goal)
            return task.Jump_Standard, -1, -1
        else:
            return task.VISION, k_median, x_mid
    elif goal == 3:
        return task.HALT, -1, -1
