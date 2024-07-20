''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

import cv2
import os
import winsound

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    # img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting 

    # if count == 1:
    #     winsound.Beep(600, 600)
    #     print("Look straight")
    # elif count <= 15:
    #     print("capturing front face")
    # elif count == 16:
    #     winsound.Beep(600, 600)
    #     print("Look your left side")
    # elif count > 15 and count <= 30:
    #     print("capturing right side of face")
    # elif count == 31:
    #     winsound.Beep(600, 600)
    #     print("Look your right side")
    # elif count > 31 and count <= 45:
    #     print("capturing your left side of face")
    # elif count == 46:
    #     winsound.Beep(600, 600)
    #     print("Look up")
    # elif count > 46 and count <= 60:
    #     print("capturing your chin ")
    # elif count == 61:
    #     winsound.Beep(600, 600)
    #     print("Look down")
    # elif count > 61 and count <= 75:
    #     print("capturing your forehead ")
    # elif count > 75:
    #     break

    if k == 27:
        break
    elif count >= 50: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


