import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd  

# Paths to models
detection_model_path = r'C:\Users\user\OneDrive\Desktop\FYP\Development Interface\Rainy\haarcascade_files\haarcascade_frontalface_default.xml'
#detection_model_path = r'C:\Users\user\OneDrive\Desktop\FYP\Development\Music-recommendation-system\haarcascade_files\haarcascade_eye.xml'
emotion_model_path = r'C:\Users\user\OneDrive\Desktop\FYP\Development\Music-recommendation-system\final_model.h5'

# Load models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["happy", "sad"]
#EMOTIONS = ["happiness", "sadness", "fear", "surprise", "neutral", "disgust"]

def emotion_testing():
    cap = cv2.VideoCapture(0)
    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_detection.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y+h, x:x+w]  # Cropping region of interest i.e. face area from image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = emotion_classifier.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = EMOTIONS[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis', resized_img)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return predicted_emotion

emotion_word = emotion_testing()
if emotion_word == 'sad':
    emotion_code = 0
else:
    emotion_code = 1

