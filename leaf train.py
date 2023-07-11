import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('tea_leaf_model.h5')

# Define the class labels
class_labels = ['diseased', 'healthy']

# Load the test image
img_path = '/path/to/test/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Preprocess the image
x = x / 255.0

# Predict the class label
preds = model.predict(x)
class_idx = np.argmax(preds[0])
class_label = class_labels[class_idx]

# Print the predicted class label
print('Predicted class label: ' + class_label)
