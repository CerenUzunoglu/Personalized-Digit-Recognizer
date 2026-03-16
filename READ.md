# Personalized Digit Recognizer

A simple handwritten digit recognition project built with a CNN trained on the MNIST dataset.

The model is lightly fine-tuned with a small custom handwriting dataset to better recognize a specific writing style.

## Features

- CNN-based digit classification
- Training on MNIST dataset
- Light fine-tuning with custom handwritten digits
- Interactive canvas interface using Gradio
- Digit prediction with confidence score

## Project Structure

CNN_DEMO
- train.py
- fine_tune.py
- app.py
- mnist_model.h5
- mnist_model_custom.h5
- requirements.txt
- ceren_digits/

## Run

Activate environment:

.\venv\Scripts\activate

Install requirements:

pip install -r requirements.txt

Train model:

python train.py

Fine tune:

python fine_tune.py

Run app:

python app.py