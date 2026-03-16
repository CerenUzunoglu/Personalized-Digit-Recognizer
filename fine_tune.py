import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

DATASET_PATH = "ceren_digits"

model = load_model("mnist_model.h5", compile=False)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


def preprocess_custom_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    _, img_thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img_thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = img_thresh[y:y+h, x:x+w]

    size = max(w, h) + 20
    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    square = cv2.dilate(square, np.ones((2, 2), np.uint8), iterations=1)

    square = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    square = square.astype("float32") / 255.0

    square = square.reshape(28, 28, 1)

    return square


X_custom = []
y_custom = []

for label in range(10):
    class_dir = os.path.join(DATASET_PATH, str(label))

    if not os.path.isdir(class_dir):
        continue

    for filename in os.listdir(class_dir):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(class_dir, filename)
            processed = preprocess_custom_image(file_path)

            if processed is not None:
                X_custom.append(processed)
                y_custom.append(label)

X_custom = np.array(X_custom, dtype="float32")
y_custom = np.array(y_custom)

print("Custom dataset shape:", X_custom.shape)
print("Custom labels shape:", y_custom.shape)

if len(X_custom) == 0:
    raise ValueError("Custom dataset couldn't be loaded.")

datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(X_custom)

history = model.fit(
    datagen.flow(X_custom, y_custom, batch_size=2),
    epochs=10,
    verbose=2
)

model.save("mnist_model_custom.h5")
print("Fine-tuned model saved as mnist_model_custom.h5")
