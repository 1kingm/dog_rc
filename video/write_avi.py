import numpy as np
import cv2
 
cap = cv2.VideoCapture(1)
 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
out = cv2.VideoWriter('write.avi',fourcc, 30.0, (640,480),True)
 
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
 
        cv2.imshow('frame',frame)
        out.write(frame)
 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
 
cap.release()
out.release()
cv2.destroyAllWindows()