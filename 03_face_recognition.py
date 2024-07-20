''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os 
import urllib.request

url = 'http://192.168.169.123/cam-mid.jpg'
uslink = 'http://192.168.169.123/ultrasonic'
im = None

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
# img_resp = urllib.request.urlopen(url)
# imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
# cam = cv2.imdecode(imgnp, -1)
# print(cam)
# cam.set(3, 640) # set video widht
# cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*640
minH = 0.1*480

while True:
    #####get video from esp32- use img as input##########
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)
    #####################################################
    print(img)
    #img = cv2.flip(img, -1) # Flip vertically

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

        if confidence < 100 and confidence > 65:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # if (confidence < 0): #and confidence > 65):
        #     id = "unknown"
        #     confidence = "  {0}%".format(round(100 - confidence))    
        # else:
        #     id = names[id]
        #     confidence = "  {0}%".format(round(100 - confidence))


        # print (confidence)
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
