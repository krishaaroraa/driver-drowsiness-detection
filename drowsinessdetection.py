import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import os

# Alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load trained model
model = load_model('models/cnncat2.h5')

# Labels
lbl = ['Closed', 'Open']
score = 0
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    rpred = [1]
    lpred = [1]

    for (x, y, w, h) in right_eye:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (24, 24)) / 255.0
        eye = eye.reshape(1, 24, 24, 1)
        rpred = model.predict(eye).argmax(axis=1)
        break

    for (x, y, w, h) in left_eye:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (24, 24)) / 255.0
        eye = eye.reshape(1, 24, 24, 1)
        lpred = model.predict(eye).argmax(axis=1)
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (0, 0, 255), 1)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (0, 255, 0), 1)

    if score < 0:
        score = 0
    if score > 15:
        cv2.imwrite("drowsy.jpg", frame)
        try:
            sound.play()
        except:
            pass
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 4)

    cv2.putText(frame, f'Score: {score}', (100, height - 20), font, 1, (255, 255, 255), 1)
    cv2.imshow('Drowsiness Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
