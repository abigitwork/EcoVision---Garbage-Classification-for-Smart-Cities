import cv2
import numpy as np
import tensorflow as tf
import argparse
import os

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}.")
        return

    # Preprocess image
    # MobileNetV2 expects 224x224 and RGB format
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (224, 224))
    
    # Scale pixels to [0, 1] as done during training
    input_img = resized_img / 255.0
    input_img = np.expand_dims(input_img, axis=0) # Add batch dimension

    # Predict
    prediction = model.predict(input_img)[0][0]
    
    # Since we mapped: 0 -> biodegradable, 1 -> non-biodegradable (based on ImageDataGenerator binary mode, let's verify alphabetical order)
    # Actually, flow_from_dataframe 'binary' assigns 0 and 1 alphabetically.
    # 'biodegradable' is 0, 'non-biodegradable' is 1.
    if prediction < 0.5:
        label = "Biodegradable"
        confidence = (1 - prediction) * 100
        color = (0, 255, 0) # Green for biodegradable
    else:
        label = "Non-biodegradable"
        confidence = prediction * 100
        color = (0, 0, 255) # Red for non-biodegradable

    # Draw label and confidence on image
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display image
    cv2.imshow("Garbage Classification", img)
    print(f"Prediction: {label} ({confidence:.2f}%)")
    
    # Wait for 3 seconds and close window
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Garbage Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, default="garbage_classifier_model.h5", help="Path to the trained model")
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    predict_image(model, args.image)
