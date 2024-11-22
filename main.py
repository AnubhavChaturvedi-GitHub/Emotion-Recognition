import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Class labels
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to exit the webcam feed.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video. Exiting...")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for (x, y, w, h) in faces:
        # Crop and preprocess the face
        face = gray_frame[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (48, 48)) / 255.0
        reshaped_face = np.reshape(resized_face, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(reshaped_face)
        emotion_label = emotion_classes[np.argmax(prediction)]

        # Display the emotion label and draw a bounding box
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Emotion Detection', frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
