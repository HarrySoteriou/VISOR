import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture(0)
quinso = cv2.imread('Quinso.png')
while True:
    r, frame = cap.read()
    cv2.flip(frame, 1)
    if r:
        
        frame = cv2.resize(frame,(763, 429)) # Downscale to improve frame rate
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        rects, weights = hog.detectMultiScale(gray)
        
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        numpy_vertical = np.vstack((quinso, frame))
        cv2.imshow('Quinso sitNL people recognision2', numpy_vertical)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break