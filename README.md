# EcoVision---Garbage-Classification-for-Smart-Cities
This project is a deep learning computer vision system designed to detect and classify waste items into **Biodegradable** and **Non-biodegradable** categories from images. It uses the TrashNet dataset, Transfer Learning with MobileNetV2, and OpenCV for inference visualization.
## Installation

1. Make sure you have Python installed.
2. Install the necessary dependencies using the `requirements.txt` file:

pip install -r requirements.txt

3. Download the dataset from: https://www.kaggle.com/datasets/feyzazkefe/trashnet
   Extract the dataset to the main folder.


## Usage

### 1. Training the Model
To start the training process, run the `train.py` script. This script will automatically load the dataset, map the classes, and begin training the MobileNetV2 model.

python train.py

- The best performing model will be saved as `garbage_classifier_model.h5`.
- A plot of the training history (loss and accuracy) will be generated as `training_history.png`.

### 2. Running Inference (Testing)
Once the model is trained, you can test it on any image using the `inference.py` script via the command line.

python inference.py --image path\to\your\image.jpg

**Example**:

python inference.py --image "dataset-resized\paper\paper1.jpg"

An OpenCV window will appear showing your image along with the predicted label and confidence score. The window will automatically close after 3 seconds.

Alternatively you can use Run this on Streamlit using the command:

streamlit run app.py

In streamlit, you can directly select the image folder or select individual images according to your choice.

## Technologies Used
- **Python**
- **TensorFlow & Keras** (Deep Learning & Transfer Learning)
- **OpenCV** (Image manipulation and visualization)
- **Pandas** (Dataframe management for ImageDataGenerator)


