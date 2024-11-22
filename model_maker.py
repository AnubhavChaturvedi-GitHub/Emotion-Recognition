import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess dataset
def load_fer2013_data(csv_path):
    data = pd.read_csv(csv_path)
    pixels = data['pixels'].tolist()
    images = np.array([np.fromstring(p, sep=' ') for p in pixels], dtype=np.float32)
    images = images.reshape(-1, 48, 48, 1) / 255.0  # Normalize pixel values
    labels = to_categorical(data['emotion'], num_classes=7)
    return images, labels

# Load data
csv_path = 'fer2013.csv'
images, labels = load_fer2013_data(csv_path)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes for emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32)

# Save the trained model
model.save('emotion_model.h5')  # Save the model to a file

print("Model has been saved as 'emotion_model.h5'.")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
