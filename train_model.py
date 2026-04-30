import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = []
labels = []

dataset_path = "dataset"

# Load face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1.0, (300,300),
                                     (104.0,177.0,123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence > 0.5:
                box = detections[0,0,i,3:7] * [w,h,w,h]
                (x1,y1,x2,y2) = box.astype("int")

                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                # ✅ IMPORTANT: convert to grayscale
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                face = cv2.resize(face, (100,100))
                data.append(face.flatten())
                labels.append(person)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(data, labels)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Training complete! Model saved as model.pkl")
