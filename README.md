# Real-Time Emotion Detection Using Deep Learning

## Project Overview

This project implements an AI-driven emotion detection system capable of recognizing five emotions: Angry, Happy, Neutral, Sad, and Surprise. It uses a fine-tuned MobileNet architecture for real-time emotion recognition from facial expressions, integrating OpenCV for face detection and a deep learning model for classification.The application is designed for practical use cases like human-computer interaction, mental health monitoring, and intelligent systems.

## How It Works

**Data Preparation:**

The model is trained on the FER-2013 dataset, which contains labeled facial expression images.
Extensive data augmentation is applied to improve generalization.

**Model Training:**

MobileNet is employed as a feature extractor, with a custom classification head for emotion detection.
The model is trained with callbacks for early stopping, learning rate reduction, and model checkpointing to ensure efficiency.

**Real-Time Emotion Detection:**

The system captures video input, detects faces, preprocesses them, and passes them to the model for emotion prediction.
The detected emotion is displayed in real time.

## Technical Description

**Model Architecture:**

Pre-trained MobileNet serves as the backbone for lightweight feature extraction.
Additional dense layers classify emotions into the predefined categories.

**Face Detection:**

OpenCV's Haar cascades or DNN-based face detection identifies regions of interest (ROI) in video frames.

**Preprocessing:**

Detected faces are resized to 224x224 pixels and normalized for model input.

**Training Details:**

Dataset: FER-2013

Optimizer: Adam

Loss function: Categorical Cross-Entropy

Callbacks: Early stopping, learning rate scheduler, and model checkpointing

**Performance:**

The model is designed for high inference speed, making it suitable for real-time applications.

**Required Libraries**

Install the following Python libraries before running the project:

```bash

pip install tensorflow keras opencv-python numpy matplotlib
```

## Usage Instructions

**Training:**

Ensure the FER-2013 dataset is available in the proper directory structure.
Run the training script to train the model. The best model will be saved as emotion_model.h5.

**Real-Time Emotion Detection:**

Run the emotion detection script to start the application.
The system will access your webcam, detect faces, and display the predicted emotion in real time.

**Testing with Static Images:**

Use the testing script to classify emotions in static images.
The detected emotion will be displayed alongside the input image.


Kaggle Dataset :- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data.

Use train.py file to train the model.

Execute the test.py file to run the Emotion Detection.
