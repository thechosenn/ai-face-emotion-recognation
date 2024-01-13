# AI Face Emotion Recognition

## Project Overview

This project aims to recognize human emotions through facial expressions using a deep learning approach. Built with Python, it leverages powerful libraries like NumPy, Keras, OpenCV, and TensorFlow to process and analyze images efficiently.

## Dataset

The project utilizes the `fer2013` dataset, which comprises 35,887 grayscale images of human faces, each sized at 48x48 pixels. These images are categorized into 5 classes of emotions:

- 0 = Angry
- 1 = Happy
- 2 = Neutral
- 3 = Sad
- 4 = Surprise

The dataset is divided into:

- Training set: 28,709 images
- Public test set: 3,178 images
- Private test set: 3,351 images

## Model Architecture

The core of this project is the MobileNetV2 model, used as the base model. Additional fully connected layers are added to tailor the model to our specific task. The output layer uses a softmax activation function to categorize emotions.

Key aspects of the model include:

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Performance**:
  - Accuracy on the public test set: 64.68%
  - Accuracy on the private test set: 65.13%

## Requirements

This project uses several key Python libraries for machine learning and image processing. To install these, you will need the following:

- NumPy: A fundamental package for scientific computing in Python.
- Keras: A deep learning API written in Python, running on top of the machine learning platform TensorFlow.
- OpenCV (Open Source Computer Vision Library): A library of programming functions mainly aimed at real-time computer vision.
- TensorFlow: An end-to-end open-source platform for machine learning.

To install all required libraries, run the following command:

To install the required libraries, run:

```bash
pip install -r requirements.txt
