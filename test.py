import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import ImageGrab
 
path = 'Images'
images = []
Names = []
myList = os.listdir(path)
for img in myList:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    Names.append(os.path.splitext(img)[0])
 
def findEncodings(images):
    encodeList = []
    c = 0
    for img in images:
        c = c+1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except:
            print(c)
    return encodeList
 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'n{name},{dtString}')
 

def captureScreen(bbox=(300,300,690+300,530+300)):
     capScr = np.array(ImageGrab.grab(bbox))
     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
     return capScr
 
encodeListKnown = findEncodings(images)
 
cap = cv2.VideoCapture(0)
while True:
    img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)   
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = Names[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
        else:
            name = 'Unknown Student'
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
 
cv2.imshow('Camera',img)
cv2.waitKey(1)