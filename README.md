# Emotion Analyser

This project implements a Convolutional Neural Network (CNN) for real-time emotion detection from facial images. The model is capable of classifying 7 different human emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Features

-   **Model Training**: Train a deep learning model (CNN) using a custom dataset of facial expressions.
-   **Real-time Detection**: Perform live emotion detection using your webcam.
-   **Emotion Classification**: Categorize emotions into 7 distinct classes.

## Installation

Follow these steps to set up the project locally:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/emotion_analyser.git
    cd emotion_analyser
    ```

2.  **Create and activate a Python virtual environment**:
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Dataset

The project expects the dataset to be organized in the `assets/images/` directory with `train` and `validation` subdirectories, each containing folders for the 7 emotion classes. For example:

```
assets/
└── images/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   └── ...
    └── validation/
        ├── angry/
        ├── disgust/
        └── ...
```

### Training the Model

To train the emotion detection model, run the `trainmodel.py` script. You can specify the training and validation directories, number of epochs, and batch size.

```bash
python trainmodel.py --train_dir assets/images/images/train --test_dir assets/images/images/validation --epochs 50 --batch_size 128
```

The trained model will be saved as `emotiondetector.json` (model architecture) and `emotiondetector.h5` (model weights). The training history will be saved in `history.json`.

### Real-time Emotion Detection

To perform real-time emotion detection using your webcam, execute the `realtimedetection.py` script:

```bash
python realtimedetection.py
```

This will open a webcam feed, detect faces, and display the predicted emotion on the detected face.

### Plotting Training History

You can visualize the training accuracy and loss from a previously saved history file:

```bash
python trainmodel.py --plot_history history.json
```

## File Structure

-   `emotiondetector.h5`: Trained model weights.
-   `emotiondetector.json`: Model architecture in JSON format.
-   `history.json`: Training history (accuracy and loss per epoch).
-   `README.md`: Project README file.
-   `realtimedetection.py`: Script for real-time emotion detection using a webcam.
-   `requirements.txt`: Python dependencies.
-   `trainmodel.py`: Script for training the emotion detection model.
-   `assets/`: Directory containing the image datasets.
    -   `images/`: Contains `train/` and `validation/` subdirectories with emotion-categorized images.