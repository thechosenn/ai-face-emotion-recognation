import tensorflow as tf 
import cv2
import os
import matplotlib.pyplot as plt 
import numpy as np
import random
from tensorflow import keras
from keras import layers

# img_array = cv2.imread('Training/train/angry/Training_3908.jpg')
# print(img_array.shape)
# plt.imshow(img_array)
# plt.show()

Datadirectory = "Training/train/"
#Classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]
Classes = ["0","1","2","3","4"]

img_size = 224 
# new_array = cv2.resize(img_array, (img_size, img_size))
# plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
# plt.show()

training_Data = []

def create_training_Data():
  for category in Classes:
    path = os.path.join(Datadirectory, category)
    class_num = Classes.index(category)
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path, img))
        new_array = cv2.resize(img_array, (img_size, img_size))
        training_Data.append([new_array, class_num])
      except Exception as e:
        pass

create_training_Data()

random.shuffle(training_Data)

X = []
y = []
for features, label in training_Data:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3) 
# 3 is the channel for RGB

X =X/255.0

model = tf.keras.applications.MobileNetV2()
base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output) 
final_output = layers.Activation('relu')(final_output) 
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)

new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

new_model.fit(X, Y, epochs = 25)

new_model.save('Final_model.h5')

