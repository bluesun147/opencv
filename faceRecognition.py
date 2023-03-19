import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# https://jinho-study.tistory.com/229?category=926937

# 얼굴 탐지할 이미지 불러옴

image = cv.imread('faces.jpeg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

xml = 'haarcascade_frontalface_default.xml'
face_cascase = cv.CascadeClassifier(xml)
faces = face_cascase.detectMultiScale(gray, 1.2, 5)

print(str(len(faces)) + "명 detected!")

if len(faces):
    for (x, y, w, h) in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB), cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()