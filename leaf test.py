import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('tea_leaf_model.h5')

# Define the class labels
class_labels = ['diseased', 'healthy']

# Open the camera
cap = cv2.VideoCapture(0)

# Set the camera resolution
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict the class label
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    class_label = class_labels[class_idx]

    # Add the predicted class label to the frame
    cv2.putText(frame, class_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Tea Leaf Disease Detector', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
