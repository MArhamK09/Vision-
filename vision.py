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

def waitfive(label, frame):
    global a
    if "person" in label:  # Check if "person" is detected
        if a == 0:
            # Perform face recognition here
            # Example:
            # Assuming you have a function called recognize_face
            # that takes the frame as input and returns True if a face is recognized
            
            if recognize_faces(frame): # we will use arham bhaiya method of live face recgntin
                engine.say("Hello, I recognize you!")
            else:
                engine.say("An unknown person is detected")
            engine.runAndWait()
        a += 1
        if a == 5:
            a = 0
    else:
        if a == 0:
            engine.say("No person detected")
            engine.runAndWait()
        a += 1
        if a == 5:
            a = 0


def recognize_faces(img):
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
    id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', "Mohammad Arham Khan",'Shashi Ranjan',  "Shreshth Jha"] 

# Initialize and start realtime video capture
    # cam = cv2.VideoCapture(0)
    # cam.set(3, 640) # set video widht
    # cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
    # minW = 0.1*cam.get(3)
    # minH = 0.1*cam.get(4)

######################################################
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
    gray,
    scaleFactor = 1.2,
    minNeighbors = 5,
    minSize = (int(3), int(4)),
    )

    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # print (confidence)
            # Check if confidence is less than 100 ==> "0" is perfect match 

        if (confidence < 100): #and confidence > 65):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

######################################################
    # while True:

    #     # ret, img =cam.read()
    # #img = cv2.flip(img, -1) # Flip vertically

    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #     faces = faceCascade.detectMultiScale( 
    #         gray,
    #         scaleFactor = 1.2,
    #         minNeighbors = 5,
    #         minSize = (int(minW), int(minH)),
    #     )

    #     for(x,y,w,h) in faces:

    #         cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    #         id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    #     # print (confidence)
    #     # Check if confidence is less than 100 ==> "0" is perfect match 

    #         if (confidence < 100): #and confidence > 65):
    #             id = names[id]
    #             confidence = "  {0}%".format(round(100 - confidence))
            
    #         else:
    #             id = "unknown"
    #             confidence = "  {0}%".format(round(100 - confidence))

    #     # if (confidence < 0): #and confidence > 65):
    #     #     id = "unknown"
    #     #     confidence = "  {0}%".format(round(100 - confidence))    
    #     # else:
    #     #     id = names[id]
    #     #     confidence = "  {0}%".format(round(100 - confidence))


    #     # print (confidence)
    #         cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
    #         cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    #     cv2.imshow('camera',img) 

    #     k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    #     if k == 27:
    #         break

# Do a bit of cleanup
    # print("\n [INFO] Exiting Program and cleanup stuff")
    # cam.release()
    # cv2.destroyAllWindows()


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
        ####################################
        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.4, model='yolov4-tiny')
        out = draw_bbox(frame, bbox, label, conf)
        steps = int(us_val//30.48)-1
        if steps<0:
            steps = 0
        if label and us_val<500:
            print(label[0], steps)
            waitfive(label[0]+" is "+str(steps)+" steps away", out)
        elif label:
            print(label[0])
            waitfive(label[0] + " is detected", out)
        elif us_val>0:
            print(steps)
            waitfive(" obstacle ahead", out)

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