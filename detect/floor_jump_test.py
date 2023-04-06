import cv2 as cv
import numpy as np

cap = cv.VideoCapture('floor_jump_test.MP4')

frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

lower_red = np.array([0, 182, 95])
upper_red = np.array([7, 255, 255])

kernel = np.ones((3, 3), dtype=np.uint8)

isVerticleStop = 0
isHorizontalStop = 0

while (cap.isOpened()):
    width = int(frame_width / 5)
    height = int(frame_height / 5)

    death_zone = 10  # 防止斜率突变的死区，需要改变
    top_idx = 0
    top = height
    area = []

    ret, img = cap.read()

    img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)

    half_height = height / 2
    half_width = width / 2
    img_rgb = cv.GaussianBlur(img, (5, 5), 0)  # 5*5 gaussian blur
    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)  # BGR to HSV
    mask_red_1 = cv.inRange(img_hsv, lower_red, upper_red)
    mask_red_2 = cv.GaussianBlur(mask_red_1, (3, 3), 0)

    cv.imshow('floor_mask', mask_red_2)
    cv.imshow('floor_frame', img)

    contours, hierarchy = cv.findContours(mask_red_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for k in range(len(contours)):
        (x, y, w, h) = cv.boundingRect(contours[k])
        if w > 30 and h > 10:
            if y < top:
                top = y
                top_idx = k

    (x, y, w, h) = cv.boundingRect(contours[top_idx])
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1, cv.LINE_AA)

    stop_num = y + h
    turn_num = x + w / 2

    if isVerticleStop == 0:
        if stop_num <= 225 :
            print("not yet")
        else:
            print("stop")
            isVerticleStop = 1

    if isVerticleStop == 1 and isHorizontalStop == 0:
        if half_width - death_zone <= turn_num <= half_width + death_zone:
            print("ready to jump")
            isHorizontalStop = 1
        elif half_width - death_zone > turn_num:
            print("turn left")
        elif half_width + death_zone > turn_num:
            print("turn right")

    cv.imshow('contours', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()