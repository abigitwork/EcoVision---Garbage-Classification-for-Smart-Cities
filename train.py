import os
import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define dataset path
DATASET_PATH = 'dataset-resized'

# Define the mapping for classes
CLASS_MAPPING = {
    'cardboard': 'biodegradable',
    'paper': 'biodegradable',
    'glass': 'non-biodegradable',
    'metal': 'non-biodegradable',
    'plastic': 'non-biodegradable',
    'trash': 'non-biodegradable'
}

def create_dataframe(dataset_path):
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
                    
    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })
    return df

def main():
    print("Creating dataframe from dataset...")
    df = create_dataframe(DATASET_PATH)
    print(f"Total images found: {len(df)}")
    print(df['label'].value_counts())

    # Data Augmentation & Loading
    print("\nSetting up ImageDataGenerators...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2, # 20% for validation
        rotation_range=30, # Increased rotation
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2], # Added brightness
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepath',
        y_col='label',
        target_size=(224, 224),
        class_mode='binary',
        batch_size=32,
        subset='training'
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepath',
        y_col='label',
        target_size=(224, 224),
        class_mode='binary',
        batch_size=32,
        subset='validation'
    )

    # Note down class indices
    print("Class indices:", train_generator.class_indices)

    # ==========================================
    # PHASE 1: Train Top Layers
    # ==========================================
    print("\nBuilding model (Phase 1: Feature Extraction)...")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze base model initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x) # Increased dense capacity
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint('garbage_classifier_model.h5', save_best_only=True, monitor='val_loss')

    # Train Model
    print("\nStarting Phase 1 training...")
    history1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30, # Increased epochs
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    # ==========================================
    # PHASE 2: Fine-Tuning
    # ==========================================
    print("\nStarting Phase 2: Fine-Tuning...")
    base_model.trainable = True
    
    # Freeze all layers except the top 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    # Recompile with a very low learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks for fine-tuning
    early_stop_ft = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr_ft = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    
    history2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[early_stop_ft, reduce_lr_ft, checkpoint]
    )

    print("\nTraining completed! Model saved as 'garbage_classifier_model.h5'.")

    # Combine histories for plotting
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='train_accuracy')
    plt.plot(val_acc, label='val_accuracy')
    plt.legend()
    plt.title('Accuracy (Phase 1 & 2)')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.title('Loss (Phase 1 & 2)')
    
    plt.savefig('training_history.png')
    print("Saved training history plot as 'training_history.png'.")

if __name__ == "__main__":
    main()
