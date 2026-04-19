import cv2
import numpy as np
import os
import time # Import the time module for debouncing
from keras.models import load_model

# Load trained model
# Ensure the path to your model is correct.
model = load_model(r"P:\Coding files\plastic\predictWaste12.h5")

# Class labels
class_names = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]
IMG_SIZE = 224

# Create output folder for saved images
os.makedirs("captures", exist_ok=True)
capture_count = 1

# --- Debouncing variables for both console and visual display ---
# This dictionary will store the last time each confident class was detected.
# Key: class name (e.g., 'Plastic'), Value: timestamp (time.time())
last_confident_detection_time = {}
# Cooldown period in seconds. Adjust this value as needed.
COOLDOWN_PERIOD = 3 # seconds

# Variables to hold the prediction that is currently being displayed on the frame
# These will only update when the debouncing condition is met.
current_display_label = "Unknown"
current_display_conf = 0.0
last_displayed_update_time = 0.0 # Time when current_display_label was last updated

# Start webcam
cap = cv2.VideoCapture("http://192.0.0.4:8888/video")

if not cap.isOpened():
    print("❌ Error: Could not open webcam. Make sure it's not in use by another application.")
    exit()

print("✅ Waste Detector Started!")
print("   - Press 'q' to quit")
print("   - Press 's' to save the current frame with its prediction")
print(f"   - Visual and console output debouncing is active: Detections will not repeat within a {COOLDOWN_PERIOD}-second window.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame. Exiting...")
        break

    # Get frame dimensions and calculate center
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    # Define the cropping area for the model input
    left = cx - IMG_SIZE // 2
    top = cy - IMG_SIZE // 2
    right = cx + IMG_SIZE // 2
    bottom = cy + IMG_SIZE // 2

    # Ensure the cropping coordinates are within frame boundaries
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    cropped = frame[top:bottom, left:right]

    # Skip prediction if the cropped area is not the expected size
    if cropped.shape[0] != IMG_SIZE or cropped.shape[1] != IMG_SIZE:
        cv2.putText(frame, "Adjust camera/window (Crop area too small)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Resize the frame to 640x480 before showing
        resized_frame = cv2.resize(frame, (640, 480))
        cv2.imshow('♻ Waste Detector - Press s to capture, q to quit', resized_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("⚠ Cannot save: Cropped area is not valid for prediction.")
        continue # Skip to the next frame if crop is invalid

    # Preprocess image for the model
    img = cropped / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction using the loaded model
    pred = model.predict(img, verbose=0)
    conf = np.max(pred)
    label_id = np.argmax(pred)
    label = class_names[label_id]

    # --- Debouncing Logic for both console and visual display ---
    current_time = time.time()

    # Check if the current prediction is confident enough
    if conf > 0.4:
        # Check if this is a new confident detection or if cooldown has passed for this label
        if label not in last_confident_detection_time or \
           (current_time - last_confident_detection_time[label]) > COOLDOWN_PERIOD:
            # This is a "new" confident detection (or enough time passed)
            print(f"Detected: {label} ({conf * 100:.1f}%)")
            last_confident_detection_time[label] = current_time # Update console debounce time

            # Update the visually displayed prediction
            current_display_label = label
            current_display_conf = conf
            last_displayed_update_time = current_time
    else:
        # If the current prediction is not confident, check if the displayed label
        # has been shown for longer than the cooldown period. If so, revert to "Unknown".
        if current_display_label != "Unknown" and \
           (current_time - last_displayed_update_time) > COOLDOWN_PERIOD:
            current_display_label = "Unknown"
            current_display_conf = 0.0
            # No need to update last_displayed_update_time here, it will be updated
            # when a new confident detection occurs.

    # Prepare text and color for drawing based on the debounced display variables
    display_text = f"{current_display_label} ({current_display_conf * 100:.1f}%)"
    display_color = (0, 255, 0) if current_display_label != "Unknown" else (0, 0, 255)

    # Draw the bounding box and prediction text on the original frame
    cv2.rectangle(frame, (left, top), (right, bottom), display_color, 2)
    cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, display_color, 2)

    resized_frame = cv2.resize(frame, (640, 480))
    cv2.imshow('♻ Waste Detector - Press s to capture, q to quit', resized_frame)
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save frame with the currently displayed label
        filename = f"captures/{current_display_label}_{capture_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")
        capture_count += 1

# Release webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()