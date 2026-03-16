import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

base_model = load_model("mnist_model.h5")
custom_model = load_model("mnist_model_custom.h5")

def preprocess_canvas_image(img):

    if isinstance(img, dict):
        img = img.get("composite", None)

    if img is None:
        return None

    img = np.array(img)

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.bitwise_not(img)

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

    square = square.reshape(1, 28, 28, 1)

    return square

def predict_digit(img):
    processed = preprocess_canvas_image(img)

    if processed is None:
        return "Çizim algılanmadı"

    base_pred = base_model.predict(processed, verbose=0)[0]
    custom_pred = custom_model.predict(processed, verbose=0)[0]

    final_pred = 0.85 * base_pred + 0.15 * custom_pred

    digit = int(np.argmax(final_pred))
    confidence = float(np.max(final_pred))


    return f"Prediction: {digit} | Confidence: {confidence:.4f}"

demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy"),
    outputs="text",
    title="Personalized Digit Recognizer",
    description="A CNN-based MNIST digit recognizer with light fine-tuning for personal handwriting."
)

demo.launch(inbrowser=True)
