import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import os
warnings.filterwarnings('ignore')

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_models', 'mobile_model.h5')

def process_mobile(image_path):
    """Process image using MobileNetV2 model"""
    try:
        # Check if image path exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return [], None

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image: {image_path}")
            return [], None
            
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        # Load model
        try:
            model = tf.keras.models.load_model('models/trained_models/mobile_model.h5')
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return [], img_rgb
            
        # Make prediction
        try:
            prediction = model.predict(np.expand_dims(img_normalized, axis=0))
            confidence = float(prediction[0][0])  # Assuming binary classification
            
            # Draw bounding box if confidence is high enough
            if confidence >= 0.5:
                height, width = img_rgb.shape[:2]
                box_width = width // 4
                box_height = height // 4
                x = (width - box_width) // 2
                y = (height - box_height) // 2
                
                # Draw blue bounding box
                cv2.rectangle(img_rgb, (x, y), (x + box_width, y + box_height), (0, 0, 255), 2)
                
                # Add confidence label
                label = f"Waste: {confidence:.1%}"
                cv2.putText(img_rgb, label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                return [{'class': 'Waste', 'confidence': confidence, 'box': (x, y, box_width, box_height)}], img_rgb
            else:
                return [{'class': 'No Waste', 'confidence': 1 - confidence, 'box': None}], img_rgb
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return [], img_rgb
            
    except Exception as e:
        print(f"Error in process_mobile: {str(e)}")
        return [], None 