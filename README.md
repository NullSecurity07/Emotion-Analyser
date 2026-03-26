# Emotion Detection from Facial Expressions

## Description

This project aims to build a model that can detect emotions from facial expressions. The model is a Convolutional Neural Network (CNN) built with Keras and TensorFlow. The model is trained on the FER2013 dataset, which contains images of faces with seven different emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Data

The training and testing images are not included in this repository. You should create the following directory structure and place the images in the corresponding directories:
```
assets/
└── images/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
```

**Important:** The `train` and `test` directories must contain subdirectories for each emotion, and each subdirectory must contain the corresponding images.

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/emotion-detection.git
```
2. Install the dependencies:
```
pip install -r requirements.txt
```

**Note:** You might see some warnings about the protobuf library when running the script. These warnings can be safely ignored.

## Usage

To train the model, run the following command:
```
python trainmodel.py
```

To predict the emotion of a single image, run the following command:
```
python trainmodel.py --predict_image path/to/your/image.jpg
```

To plot the training history, run the following command:
```
python trainmodel.py --plot_history history.json
```

To detect emotions in real-time using a webcam, run the following command:
```
python realtimedetection.py
```

## File Descriptions

- `trainmodel.py`: The main script to train and evaluate the model.
- `realtimedetection.py`: A script to detect emotions in real-time using a webcam.
- `requirements.txt`: A file containing the required dependencies.
- `emotiondetector.json`: The model architecture in JSON format.
- `emotiondetector.h5`: The trained model weights.
- `history.json`: The training history.
- `assets/`: A directory containing the images.
