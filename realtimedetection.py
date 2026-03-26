#!/usr/bin/env python
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import argparse

def extract_features(image):
    """Extract features from a single image.

    Args:
        image (numpy.ndarray): The image to extract features from.

    Returns:
        numpy.ndarray: An array of features.
    """
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def main():
    """Detect emotions in real-time using a webcam."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="emotiondetector.json", help="The model architecture in JSON format")
    parser.add_argument("--weights", default="emotiondetector.h5", help="The trained model weights")
    parser.add_argument("--cascade", default=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', help="The cascade classifier file")
    args = parser.parse_args()

    json_file = open(args.model, "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.load_weights(args.weights)
    face_cascade=cv2.CascadeClassifier(args.cascade)

    webcam=cv2.VideoCapture(0)
    labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
    while True:
        i,im=webcam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(im,1.3,5)
        try: 
            for (p,q,r,s) in faces:
                image = gray[q:q+s,p:p+r]
                cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
                image = cv2.resize(image,(48,48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
            cv2.imshow("Output",im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            pass

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()