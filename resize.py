import os
import cv2
import numpy as np
import pandas as pd

# Load labels
legend = pd.read_csv("legend.csv")
legend["emotion"] = legend["emotion"].str.lower()

# Path to images
image_folder = "images/"

# Choose image size
IMAGE_SIZE = 32

X = []
y = []

missing = 0

for idx, row in legend.iterrows():
    img_name = row["image"]
    emotion = row["emotion"]
    path = os.path.join(image_folder, img_name)

    if not os.path.exists(path):
        missing += 1
        continue

    # Load & resize
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Flatten
    img = img.flatten()

    X.append(img)
    y.append(emotion)

print("Loaded images:", len(X))
print("Missing images:", missing)

X = np.array(X)
y = np.array(y)
