import cv2

# Make sure the path to the HAAR cascade file is correct
haar_cascade_path = 'haarcascade_frontalface_default.xml'

# Load the cascade
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Check if the cascade file is loaded correctly
if face_cascade.empty():
    print("Error: Could not load HAAR cascade file. Check the path.")
else:
    # Read the input image
    img_path = 'gp.jpg'  # Update this to the correct path of your image
    img = cv2.imread(img_path)

    # Check if the image is loaded correctly
    if img is None:
        print(f"Error: Could not load image at {img_path}. Check the path.")
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the output
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
import cv2  # opencv-python
import numpy as np  # numpy
import tensorflow as tf  # tensorflow
from tensorflow import keras  # keras
import cv2
import os

def capture_hand_gestures(num_images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Hand Gesture', frame)
        file_path = os.path.join(save_dir, f'gesture_{count}.jpg')
        cv2.imwrite(file_path, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_hand_gestures(100, 'hand_gestures')
import tensorflow as tf
tf.keras.applications.vgg16.preprocess_input(
    x= 5, data_format=None
)
i = keras.layers.Input([None, None, 3], dtype="uint8")
x = ops.cast(i, "float32")
x = keras.applications.mobilenet.preprocess_input(x)
core = keras.applications.MobileNet()
x = core(x)
model = keras.Model(inputs=[i], outputs=[x])
result = model(image)
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load VGG16 model without the top layer
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = model.predict(image, verbose=0)
    return features

# Extract features for all images
features = []
labels = []

gesture_dir = 'hand_gestures'
for img_name in os.listdir(gesture_dir):
    img_path = os.path.join(gesture_dir, img_name)
    feature = extract_features(img_path, model)
    features.append(feature)
    labels.append(img_name.split('_')[1])  # Assuming the label is part of the file name

features = np.array(features).reshape(len(features), -1)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(features.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
def classify_hand_gesture(image, model, classifier_model):
    feature = extract_features(image, model)
    feature = feature.reshape(1, -1)
    prediction = classifier_model.predict(feature)
    return le.inverse_transform([np.argmax(prediction)])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gesture = classify_hand_gesture(frame, model, model)
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2

def capture_images(num_images=100):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Images")
    img_counter = 0

    while img_counter < num_images:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Images", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            img_name = f"hand_image_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

capture_images()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(image_directory):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        image_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training')
    validation_generator = datagen.flow_from_directory(
        image_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation')
    return train_generator, validation_generator

train_gen, val_gen = preprocess_images('path_to_images')
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, TimeDistributed

model = Sequential([
    base_model,
    TimeDistributed(Flatten()),
    LSTM(256, return_sequences=False),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)
import numpy as np
import numpy as np

def live_hand_recognition(model):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Recognition")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        label = 'Hand Detected' if prediction > 0.5 else 'No Hand'

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Recognition", frame)

        if cv2.waitKey(1) % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

live_hand_recognition(model)
