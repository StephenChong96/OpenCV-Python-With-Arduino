import numpy as np
import cv2
import pickle

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

face_cascade = cv2.CascadeClassifier('Haar Cascade Classifier\data\haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainedface.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
make_1080p()

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors= 5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y: y+h, x: x+w] # Region of interest
        roi_color = frame[y: y+h, x: x+w] # (ycord_start, ycord_end)
        
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
        
        img = 'my image.png'
        cv2.imwrite(img, roi_gray)

        # Draw rectangle
        color = (255, 255, 0) # BGR 0-255 (255 strongest)(Blue = 255)
        stroke = 2          # Line width
        end_cord_x = x+w  # Width
        end_cord_y = y+h  # Height
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('p'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#