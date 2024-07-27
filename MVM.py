import cv2
import os

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:04d}.jpg"), frame)
        count += 1
    cap.release()

# Example usage
video_path = 'path_to_video.mp4'
output_dir = 'path_to_output_frames'
os.makedirs(output_dir, exist_ok=True)
extract_frames(video_path, output_dir)
import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results

def process_frames(input_dir, output_file):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    data = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.jpg'):
            landmarks = detect_hand_landmarks(os.path.join(input_dir, filename))
            if landmarks.multi_hand_landmarks:
                for hand_landmarks in landmarks.multi_hand_landmarks:
                    hand_data = []
                    for lm in hand_landmarks.landmark:
                        hand_data.extend([lm.x, lm.y, lm.z])
                    data.append(hand_data)
    hands.close()
    return data

# Example usage
input_dir = 'path_to_output_frames'
output_file = 'hand_landmarks.csv'
landmark_data = process_frames(input_dir, output_file)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

# Replace with your actual data
X = np.array(landmark_data)  # Ensure this is a 2D array
y = np.array([0, 3])  # Correct way to create a 1D array

# Check the shape of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Ensure y is a 1D array
if y.ndim != 1:
    y = np.squeeze(y)

# Check again after squeezing
print("Shape of y after squeezing:", y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import numpy as np
from sklearn.model_selection import train_test_split

# Simulated data
landmark_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]  # Example 2D list
labels = [0, 1, 0]  # Example 1D list

# Convert to numpy arrays
X = np.array(landmark_data)
y = np.array(labels)

# Check the shapes
print("Shape of X:", X.shape)  # Should be (n_samples, n_features)
print("Shape of y:", y.shape)  # Should be (n_samples,)

# Ensure y is 1D
if y.ndim != 1:
    y = np.squeeze(y)

print("Shape of y after squeezing:", y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shapes after splitting:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)



# Train a model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
import matplotlib.pyplot as plt

def plot_movements(movement_data):
    x = range(len(movement_data))
    y = movement_data

    plt.plot(x, y, label='Hand Movements')
    plt.xlabel('Time')
    plt.ylabel('Movement')
    plt.title('Hand Movement Detection')
    plt.legend()
    plt.show()

# Example usage
movement_data = [0, 1, 0, 2, 1, 0]  # Replace with your movement data (output of the model)
plot_movements(movement_data)
import cv2
import os
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Extract frames from video
def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:04d}.jpg"), frame)
        count += 1
    cap.release()

# Detect hand landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results

def process_frames(input_dir):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    data = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.jpg'):
            landmarks = detect_hand_landmarks(os.path.join(input_dir, filename))
            if landmarks.multi_hand_landmarks:
                for hand_landmarks in landmarks.multi_hand_landmarks:
                    hand_data = []
                    for lm in hand_landmarks.landmark:
                        hand_data.extend([lm.x, lm.y, lm.z])
                    data.append(hand_data)
    hands.close()
    return data

# Train a model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    return model

# Plot movements
def plot_movements(movement_data):
    x = range(len(movement_data))
    y = movement_data
    plt.plot(x, y, label='Hand Movements')
    plt.xlabel('Time')
    plt.ylabel('Movement')
    plt.title('Hand Movement Detection')
    plt.legend()
    plt.show()

# Main
video_path = 'path_to_video.mp4'
output_dir = 'path_to_output_frames'
os.makedirs(output_dir, exist_ok=True)
extract_frames(video_path, output_dir)

input_dir = 'path_to_output_frames'
landmark_data = process_frames(input_dir)

# Replace with actual labels
labels = [...]  

X = np.array(landmark_data)
y = np.array(labels)

model = train_model(X, y)

# Replace with model predictions
movement_data = model.predict(X_test)
plot_movements(movement_data)
