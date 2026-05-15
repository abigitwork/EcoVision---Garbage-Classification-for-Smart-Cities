import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASS_MAPPING = {
    'cardboard': 'biodegradable',
    'paper': 'biodegradable',
    'glass': 'non-biodegradable',
    'metal': 'non-biodegradable',
    'plastic': 'non-biodegradable',
    'trash': 'non-biodegradable'
}

filepaths = []
labels = []
dataset_path = 'dataset-resized'
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir) and class_name in CLASS_MAPPING:
        mapped_label = CLASS_MAPPING[class_name]
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_dir, img_file))
                labels.append(mapped_label)

df = pd.DataFrame({'filepath': filepaths, 'label': labels})
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_dataframe(
    dataframe=df, x_col='filepath', y_col='label',
    target_size=(224, 224), class_mode='binary', batch_size=32
)
print("CLASS INDICES:", train_generator.class_indices)
