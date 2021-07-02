import numpy as np
import cv2
import pickle
import serial,time

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

face_cascade = cv2.CascadeClassifier('Haar Cascade Classifier\data\haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainedface.yml")
# Setup communication path for arduino (In place of 'COM5' put the port to which your arduino is connected)
arduino = serial.Serial('COM3', 9600)
time.sleep(1)

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(1)
# Resize frame to 720p
make_720p()

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)  # Face detector
    for x,y,w,h in faces:
        # Sending coordinates to Arduino
        data = 'X{0:d}Y{1:d}'.format((x+w//2), (y+h//2))
        print(data)
        arduino.write(data.encode('utf-8'))
        roi_gray = gray[y: y+h, x: x+w] # Region of interest
        # Plot the center of the face
        cv2.circle(frame,(x+w//2, y+h//2), 2, (0, 255, 0), 2)
        # Plot the roi
        cv2.rectangle(frame, (x, y),(x+w, y+h), (255, 255, 0), 3)
        # Plot the squared region in the center of the screen
        cv2.rectangle(frame,(1280//2-30, 720//2-30), (1280//2+30, 720//2+30), (255, 255, 255), 3)

        # Recognizor (Deep learned model predict: keras, tensorflov, pytorch, scikit-learn)
        id, conf = recognizer.predict(roi_gray)
        if conf >= 30:
            print(id)
            print(labels[id])
            font = cv2.QT_FONT_NORMAL
            name = labels[id]
            color = (255, 255, 255) # White
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    
  # Display the resulting frame
    cv2.imshow('frame', frame)
    # Press p to quit
    if cv2.waitKey(20) & 0xFF == ord('p'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
