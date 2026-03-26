#!/usr/bin/env python
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

def createdataframe(dir):
    """Create a dataframe with image paths and labels.

    Args:
        dir (str): The directory containing the images.

    Returns:
        tuple: A tuple containing two lists: image_paths and labels.
    """
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths,labels

def extract_features(images):
    """Extract features from a list of images.

    Args:
        images (list): A list of image paths.

    Returns:
        numpy.ndarray: An array of features.
    """
    features = []
    for image in images:
        img = load_img(image,grayscale =  True )
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features),48,48,1)
    return features

def extract_features_single(image):
    """Extract features from a single image.

    Args:
        image (str): The path to the image.

    Returns:
        numpy.ndarray: An array of features.
    """
    img = load_img(image,grayscale =  True )
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def create_model():
    """Create the model."""
    model = Sequential()
    # convolutional layers
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    # fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    # output layer
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )
    return model

def plot_prediction(model, image_path, true_label):
    """Plot the prediction for a single image.

    Args:
        model (keras.models.Sequential): The trained model.
        image_path (str): The path to the image.
        true_label (str): The true label of the image.
    """
    label = ['angry','disgust','fear','happy','neutral','sad','surprise']
    img = extract_features_single(image_path)
    pred = model.predict(img)
    pred_label = label[pred.argmax()]
    
    plt.imshow(img.reshape(48,48),cmap='gray')
    plt.title(f"Original: {true_label}, Predicted: {pred_label}")
    plt.show()

def save_history(history, filename):
    """Save the training history to a file.

    Args:
        history (keras.callbacks.History): The training history.
        filename (str): The name of the file to save the history to.
    """
    with open(filename, 'w') as f:
        json.dump(history.history, f)

def plot_history(history_file):
    """Plot the training history from a history file.

    Args:
        history_file (str): The path to the history file.
    """
    with open(history_file, 'r') as f:
        history = json.load(f)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

def main():
    """Train and evaluate the model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="assets/images/images/train", help="The training directory")
    parser.add_argument("--test_dir", default="assets/images/images/validation", help="The validation directory")
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size")
    parser.add_argument("--predict_image", help="The image to predict")
    parser.add_argument("--history_file", default="history.json", help="The file to save the training history to")
    parser.add_argument("--plot_history", help="The history file to plot")
    args = parser.parse_args()

    if args.plot_history:
        plot_history(args.plot_history)
    elif args.predict_image:
        from tensorflow.keras.models import model_from_json
        json_file = open("emotiondetector.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("emotiondetector.h5")
        plot_prediction(model, args.predict_image, "Unknown")
    else:
        train = pd.DataFrame()
        train['image'], train['label'] = createdataframe(args.train_dir)
        print(train)
        test = pd.DataFrame()
        test['image'], test['label'] = createdataframe(args.test_dir)
        print(test)
        print(test['image'])
        
        train_features = extract_features(train['image']) 
        test_features = extract_features(test['image'])
        x_train = train_features/255.0
        x_test = test_features/255.0
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(train['label'])
        y_train = le.transform(train['label'])
        y_test = le.transform(test['label'])
        y_train = to_categorical(y_train,num_classes = 7)
        y_test = to_categorical(y_test,num_classes = 7)
        model = create_model()
        history = model.fit(x= x_train,y = y_train, batch_size = args.batch_size, epochs = args.epochs, validation_data = (x_test,y_test)) 
        save_history(history, args.history_file)
        model_json = model.to_json()
        with open("emotiondetector.json",'w') as json_file:
            json_file.write(model_json)
        model.save("emotiondetector.h5")

if __name__ == '__main__':
    main()