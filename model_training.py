import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# === Load Data ===
def load_images(folders, label):
    images, labels = [], []
    for folder in folders:
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (24, 24))
                img = img / 255.0
                images.append(img)
                labels.append(label)
    return images, labels

# Folder paths
closed_folders = ['train_dataset/closedLeftEyes', 'train_dataset/closedRightEyes']
open_folders = ['train_dataset/openLeftEyes', 'train_dataset/openRightEyes']

open_imgs, open_labels = load_images(open_folders, 1)
closed_imgs, closed_labels = load_images(closed_folders, 0)

X = np.array(open_imgs + closed_imgs).reshape(-1, 24, 24, 1)
y = np.array(open_labels + closed_labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)

# === Build CNN Model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train ===
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# === Save the Model ===
os.makedirs("models", exist_ok=True)
model.save('models/cnncat2.h5')
print("Model saved to models/cnncat2.h5 ✅")
