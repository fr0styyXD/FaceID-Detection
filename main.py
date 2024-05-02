import threading
import cv2
from deepface import DeepFace

cpt = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cpt.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cpt.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ctr = 0

img = cv2.imread("ref2.jpg")

face_match = False


def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret, frame = cpt.read()
    faces = faceDetect.detectMultiScale(frame)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

    if ret:
        if ctr % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        ctr += 1
        if face_match:
            cv2.putText(frame, "Access Granted", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Access Denied", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()