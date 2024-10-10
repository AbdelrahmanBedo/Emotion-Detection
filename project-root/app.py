from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

app = Flask(__name__)

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the emotion detection model
emotion_model = Sequential()
emotion_model.add(Input(shape=(48, 48, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load model weights
emotion_model.load_weights("C:\\Users\\omara\\OneDrive\\Desktop\\project-root\\emotion_model_1.h5")

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier("C:\\Users\\omara\\OneDrive\\Desktop\\project-root\\haarcascade_frontalface_default.xml")

@app.route('/')
def index():
    return render_template('exp.html')

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.get_json()
        image_data = data['image']

        # Decode base64 image
        header, encoded = image_data.split(',', 1)
        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded)).convert('L')  # Convert to grayscale

        # Convert PIL image to OpenCV format
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Detect faces in frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        emotions = []
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
            cropped_img = cropped_img / 255.0  # Normalize

            # Predict emotion
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotions.append(emotion_dict[maxindex])

        if emotions:
            result = emotions[0]
        else:
            result = "No face detected"

        return jsonify({'result': result})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'result': 'Error processing frame'}), 500

if __name__ == '__main__':
    app.run(debug=True)
