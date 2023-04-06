import cv2
import numpy as np

video_name = 1
cap = cv2.VideoCapture(video_name)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 膨胀腐蚀操作卷积核大小
hsv_low = np.array([0,18,0])
hsv_high = np.array([255,255,255])

def about_equal(x, y, percent):
    if (x >= y*(1+percent)) & (x <= y*(1-percent)):
        return 1
    else:
        return 0

cam = cv2.VideoCapture(1)
if cam.isOpened() == 0:
    print("Can't open the camera!")
else:
    print("Camera is opening!")
    width = 400
    height = 400
    
def front_seesaw(dot3, dot4):
    if about_equal(dot3[0], dot4[0], 0.05):
        y = dot3[1] + dot4[1]
        k = 0
    else:
        y = 0
        k = (dot3[1] - dot4[1])/(dot3[0] - dot4[0])
    return [y, k]

def on_seesaw(mask):
    img_canny = cv2.Canny(mask, 40, 100)
    lines = cv2.HoughLinesP(img_canny, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)
    list_kb = []
    x_median = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        k = (x1 - x2)/(y1 - y2)
        b = x1 - k*y1
        list_kb.append([k, b])
    for y in range(1, mask.shape[0]):
        x1 = list_kb[0][0]*y + list_kb[0][1]
        x2 = list_kb[1][0]*y + list_kb[1][1]
        x_median.append((x1+x2)/2)
    # 最小二乘法拟合
    mid_param = np.polyfit(x_median, range(1, mask.shape[0]), 1)
    start_point = [0, int(np.polyval(mid_param, 0))]
    end_point = [299, int(np.polyval(mid_param, 299))]
    try:
        return (start_point[1] - end_point[1])/(start_point[0] - end_point[0])  
    except:
        return 0

while True:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, hsv_low, hsv_high)
    mask_median = cv2.medianBlur(mask, 3)
    mask_open = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)
    mask_open = 255 - mask_open
    cv2.imshow('mask', mask_open)
    contours, _hierarchy = cv2.findContours(mask_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    # try:
    for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))
    max_id = areas.index(max(areas))
    max_rect = cv2.minAreaRect(contours[max_id])  # 最小外接矩阵
    max_box = np.int0(cv2.boxPoints(max_rect)) # 四个顶点并取整数
    frame = cv2.drawContours(frame,contours[max_id],-1,(0,255,0),2) 
    frame = cv2.drawContours(frame,[max_box],0,(0,255,0),2) 
    cv2.imshow('frame', frame)
    
    # 最上面的两个点极其靠近图像最高点，则检测到了
    # 这里放一个赋值语句
    dot1, dot2, dot3, dot4 = [[max_box[0][0]-0.5*max_box[1][0],max_box[0][1]+0.5*max_box[1][1]],[max_box[0][0]-0.5*max_box[1][0],max_box[0][1]+0.5*max_box[1][1]],[max_box[0][0]-0.5*max_box[1][0],max_box[0][1]-0.5*max_box[1][1]],[max_box[0][0]+0.5*max_box[1][0],max_box[0][1]+0.5*max_box[1][1]]]
    if about_equal(dot1[0], dot2[0], 0.05) & about_equal(dot1[1], 0, 0.05) & about_equal(dot2[1], 0, 0.05):
        [y, k] = front_seesaw(dot3, dot4)
        print("front")
    elif ((not about_equal(dot1[0], dot2[0], 0.05)) & (not about_equal(dot1[1], 0, 0.05)) & (not about_equal(dot2[1], 0, 0.05))) & (about_equal(dot3[0], dot4[0], 0.05) & about_equal(dot3[1], frame.shape[0], 0.05) & about_equal(dot2[1], frame.shape[0], 0.05)):
        k = on_seesaw(mask_open)
        print("on")
    
    # except ValueError:
    #     print("未检测到轮廓")
    wait = cv2.waitKey(1)
    if wait == 27:
        break
cv2.destroyAllWindows()
cap.release()