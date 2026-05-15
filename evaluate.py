import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

model = tf.keras.models.load_model('garbage_classifier_model.h5')

CLASS_MAPPING = {
    'cardboard': 'biodegradable',
    'paper': 'biodegradable',
    'glass': 'non-biodegradable',
    'metal': 'non-biodegradable',
    'plastic': 'non-biodegradable',
    'trash': 'non-biodegradable'
}

dataset_path = 'dataset-resized'
filepaths = []
labels = []

for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir) and class_name in CLASS_MAPPING:
        mapped_label = CLASS_MAPPING[class_name]
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_dir, img_file))
                labels.append(mapped_label)

# sample 200 random images
import random
random.seed(42)
sample = random.sample(list(zip(filepaths, labels)), 200)

def predict_cv2(image_path):
    image = Image.open(image_path)
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    resized_img = cv2.resize(img_array, (224, 224))
    input_img = resized_img / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    prediction = model.predict(input_img, verbose=0)[0][0]
    return "biodegradable" if prediction < 0.5 else "non-biodegradable"

def predict_pil(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    input_img = np.array(image) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    prediction = model.predict(input_img, verbose=0)[0][0]
    return "biodegradable" if prediction < 0.5 else "non-biodegradable"

correct_cv2 = 0
correct_pil = 0
for path, label in sample:
    if predict_cv2(path) == label:
        correct_cv2 += 1
    if predict_pil(path) == label:
        correct_pil += 1

print(f"CV2 Logic Accuracy: {correct_cv2}/200 ({correct_cv2/200*100:.2f}%)")
print(f"PIL Logic Accuracy: {correct_pil}/200 ({correct_pil/200*100:.2f}%)")
