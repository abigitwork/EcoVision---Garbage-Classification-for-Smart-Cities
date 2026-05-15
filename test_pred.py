import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

model = tf.keras.models.load_model('garbage_classifier_model.h5')

def predict_image(image_path):
    image = Image.open(image_path)
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    resized_img = cv2.resize(img_array, (224, 224))
    input_img = resized_img / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    prediction = model.predict(input_img, verbose=0)[0][0]
    
    if prediction < 0.5:
        return f"{image_path}: Biodegradable (val={prediction:.4f})"
    else:
        return f"{image_path}: Non-Biodegradable (val={prediction:.4f})"

print(predict_image('dataset-resized/paper/paper1.jpg'))
print(predict_image('dataset-resized/cardboard/cardboard1.jpg'))
print(predict_image('dataset-resized/plastic/plastic1.jpg'))
print(predict_image('dataset-resized/glass/glass1.jpg'))
print(predict_image('dataset-resized/metal/metal1.jpg'))
