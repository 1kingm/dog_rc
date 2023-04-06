import cv2 as cv
import numpy as np

cap = cv.VideoCapture('red_video.MP4')

frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

lower_red = np.array([0, 62, 0])
upper_red = np.array([255, 255, 255])

while (cap.isOpened()):
  ret, img = cap.read()

  width = 500
  height = 600

  img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)

  half_height = height / 2
  half_width = width / 2
  right_line_most_left_point = width
  left_line_most_right_point = 0
  img_rgb = cv.GaussianBlur(img, (5, 5), 0)  # 5*5 gaussian blur
  img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)  # BGR to HSV
  mask = cv.inRange(img_hsv, lower_red, upper_red)
  mask = cv.GaussianBlur(mask, (3, 3), 0)

  cv.imshow('floor_mask', mask)
  cv.imshow('floor_frame', img)

  contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

  for cnt in contours:
    (x, y, w, h) = cv.boundingRect(cnt)  # xy are the coordinate of upper left corner point，w stands for width，h stand for height
    if(w > 10 and h >20):
      left_most = tuple(cnt[cnt[:, :, 0].argmin()][0]) 
      left_most_the_same = cnt[:, :, 0].min()

      if (left_most[0] > half_width and left_most[0] < right_line_most_left_point):
        right_line_most_left_point = left_most[0]

      right_most = tuple(cnt[cnt[:, :, 0].argmax()][0])
      if (right_most[0] < half_width and right_most[0] > left_line_most_right_point):
        left_line_most_right_point = right_most[0]

      top_most = tuple(cnt[cnt[:, :, 1].argmin()][0])
      bottom_most = tuple(cnt[cnt[:, :, 1].argmax()][0])

      cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1, cv.LINE_AA)
      cv.circle(img, left_most, 5, (0, 255, 0), -1, cv.LINE_AA)
      cv.circle(img, right_most, 5, (0, 255, 0), -1, cv.LINE_AA)
      # cv.circle(img, top_most, 5, (0, 255, 0), -1, cv.LINE_AA)
      # cv.circle(img, bottom_most, 5, (0, 255, 0), -1, cv.LINE_AA)

  width_mid_line = (left_line_most_right_point + right_line_most_left_point) / 2
  x_bias = half_height - width_mid_line  # bigger than 0 means we need turn right
  print(x_bias)

  if (x_bias > 0 and x_bias < 20):
    print("turn right a liitle bit")
  elif (x_bias > 20):
    print("turn right a lot")
  elif (x_bias < 0 and x_bias > -20):
    print("turn right a liitle bit")
  elif (x_bias < -20):
    print("turn left a lot")
  cv.imshow('contours', img)
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()