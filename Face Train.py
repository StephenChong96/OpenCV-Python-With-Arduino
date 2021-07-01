import os
import cv2
import numpy as np
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'Images')

face_cascade = cv2.CascadeClassifier('Haar Cascade Classifier\data\haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('PNG'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace('-', ' ').lower()
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id = label_ids[label]
            # print(label_ids)
            # y_labels.append(label)  # Some number
            # x_train.append(path)    # Verify this image, turn into NUMPY array, GRAY
            pil_image = Image.open(path).convert('L')   # Grayscale
            size = (300,300) 
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array  = np.array(pil_image, 'uint8')
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor= 1.5, minNeighbors= 5)

            for(x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)

# print(x_train)
# print(y_labels)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainedface.yml")