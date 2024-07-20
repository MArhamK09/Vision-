import cv2
import urllib.request
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
import concurrent.futures
import requests

import pyttsx3
engine = pyttsx3.init()
#####################################

url = 'http://192.168.169.123/cam-mid.jpg'
uslink = 'http://192.168.169.123/ultrasonic'
im = None
a = 0

#####################face detetction part##########################

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', "Mohammad Arham Khan",'Shashi Ranjan',  "Shreshth Jha"] 

minW = 0.1*640
minH = 0.1*480

def face_rec(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    
    for(x,y,w,h) in faces:
    
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # print (confidence)
        # Check if confidence is less than 100 ==> "0" is perfect match 

        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        ###return the name of person
        return id
    return "None"  
    

##################################################################

def waitfive(txt):
    global a
    if a == 0:
        engine.say(txt)
        engine.runAndWait()
    a += 1
    if a == 5:
        a = 0

def object_detection():
    while True:
        img_resp = urllib.request.urlopen(url)
        
        
        usresponse = requests.get(uslink)
        if usresponse.status_code == 200:
            us_val = int(usresponse.text)
        else:
            us_val = 0
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        name = face_rec(frame)
        ####################################
        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.4, model='yolov4-tiny')
        out = draw_bbox(frame, bbox, label, conf)
        steps = int(us_val//30.48)-1
        if steps<0:
            steps = 0
        if name != 'unknown' and name!='None' and  us_val<500:
            print(name, steps)
            waitfive(name+" is "+str(steps)+" steps away")
        elif name != 'unknown' and name!='None':
            print(name, steps)
            waitfive(name+" detected")
        elif name == "unknown" and name == "none" and  us_val<500:
            print("person", steps)
            waitfive("person is "+str(steps)+" steps away")
        elif label and us_val<500:
            print(label[0], steps)
            waitfive(label[0]+" is "+str(steps)+" steps away")
        elif label:
            print(label[0])
            waitfive(label[0] + " is detected")
        elif us_val>0:
            print(steps)
            waitfive(" obstacle ahead")

        cv2.imshow('object detection', out)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    # webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Object Detection Started")
    # with concurrent.futures.ProcessPoolExecutor() as executer:
    #     od = executer.submit(objectDetection)
    object_detection()