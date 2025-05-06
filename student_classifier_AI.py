import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import time

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the attendance record
try:
    attendance_df = pd.read_csv("attendance.csv")
except FileNotFoundError:
    print("Error: attendance.csv not found.")
    exit()

print("Initial Attendance DataFrame:")
print(attendance_df)

# Load the Teachable Machine model using TensorFlow Lite's interpreter
model_path = "model/model.tflite"
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
except ValueError:
    print(f"Error loading model from {model_path}")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Load class labels from labels.txt
try:
    with open("model/labels.txt", "r") as f:
        class_labels = f.read().splitlines()
except FileNotFoundError:
    print("Error: labels.txt not found.")
    exit()

print("Class Labels:")
print(class_labels)

# Start the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# To avoid redundant face recognition, keep track of recognized faces
recognized_faces = {}

def preprocess_face(gray_img, face_coords):
    x, y, w, h = face_coords
    face_img = gray_img[y:y+h, x:x+w]
    resized_img = cv2.resize(face_img, (224, 224))
    input_data = np.expand_dims(resized_img, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)  # Add a channel dimension
    input_data = np.repeat(input_data, 3, axis=-1)  # Repeat the channel for RGB (3 channels)
    input_data = input_data.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return input_data

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_data = preprocess_face(gray, (x, y, w, h))

        # Check if face is already recognized in this session
        face_key = (x, y, w, h)  # Use face coordinates as a unique key
        if face_key in recognized_faces:
            name = recognized_faces[face_key]
            print(f"Recognized (cached): {name}")
        else:
            interpreter.set_tensor(input_index, face_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)
            confidence = np.max(output_data)

            if confidence > 0.8:
                label = np.argmax(output_data)
                if label < len(class_labels):
                    class_name = class_labels[label]
                    name_row = attendance_df[attendance_df["Name"] == class_name]
                    
                    if not name_row.empty:
                        name = name_row["Name"].values[0]
                        recognized_faces[face_key] = name  # Cache the recognition result
                        print(f"Recognized: {name}")

                        # Update attendance record and mark the timestamp
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        attendance_df.loc[attendance_df["Name"] == class_name, ["Attendance", "Timestamp"]] = ["Present", timestamp]

                        # Draw a rectangle and name label around the face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, "Present", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        print(f"Name '{class_name}' not found in attendance DataFrame")
                else:
                    print("Label out of range of class labels")
            else:
                print("Face not recognized with high confidence")

    cv2.imshow('Attendance System', frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the updated attendance record
print("Updated Attendance DataFrame:")
print(attendance_df)
attendance_df.to_csv("attendance.csv", index=False)

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
