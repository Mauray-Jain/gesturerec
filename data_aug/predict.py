import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("cnn.keras")

# Parameters
REGION_SIZE = (300, 300)  # Region for hand detection
LABELS = ['back_palm', 'fist', 'front_palm', 'index_finger', 'left_thumb', 'peace', 'right_pinky']

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)

    # Define the region on the RIGHT side
    height, width, _ = frame.shape
    x1 = width - REGION_SIZE[0] - 10  # 10px padding from the right
    y1 = 10  # 10px padding from the top
    x2, y2 = x1 + REGION_SIZE[0], y1 + REGION_SIZE[1]

    # Extract ROI (Region of Interest)
    roi = frame[y1:y2, x1:x2]

    # Predict gesture
    img_array = tf.keras.utils.img_to_array(roi)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_label = LABELS[np.argmax(score)]
    confidence = np.max(score) * 100

    # Display Prediction below the ROI
    text = f"{predicted_label} ({confidence:.2f}%)"
    if confidence < 90:
        text = "None"
    cv2.putText(frame, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw region on frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show frames
    cv2.imshow("Hand Gesture Recognition", frame)
    cv2.imshow("ROI", roi)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
