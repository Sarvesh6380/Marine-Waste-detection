import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import os
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('detect_waste.log')
    ]
)

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Global model cache
loaded_models = {}

def load_model_safely(model_type='cnn'):
    """
    Safely load a model with caching
    """
    try:
        # Check if model is already loaded
        if model_type in loaded_models:
            logging.info(f"Using cached model: {model_type}")
            return loaded_models[model_type]
            
        # Get model path
        model_path = os.path.join(BASE_DIR, 'trained_models', f'{model_type}_model.h5')
        logging.info(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load model
        logging.info("Loading model...")
        
        # Create a new model from scratch as the most reliable approach
        logging.info("Creating new model from scratch")
        if model_type == 'cnn':
            from train_models import create_cnn_model
            model = create_cnn_model()
        elif model_type == 'rcnn':
            from train_models import create_rcnn_model
            model = create_rcnn_model()
        elif model_type == 'mobilenet':
            from train_models import create_mobilenet_model
            model = create_mobilenet_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Try to load weights
        try:
            logging.info("Attempting to load weights from saved model")
            model.load_weights(model_path)
            logging.info("Weights loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load weights: {str(e)}")
            logging.info("Using randomly initialized weights instead")
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("Model compiled successfully")
        
        # Cache the model
        loaded_models[model_type] = model
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

def detect_objects(image):
    """
    Detect objects in an image using OpenCV
    """
    try:
        logging.info("Starting object detection")
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Get image dimensions
        height, width = image.shape[:2]
        image_area = height * width
        
        # Filter contours
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = (w * h) / image_area
            
            if 0.001 <= area_ratio <= 0.9:  # Filter by area
                bboxes.append([x, y, x + w, y + h])
        
        logging.info(f"Found {len(bboxes)} potential objects")
        return bboxes
    except Exception as e:
        logging.error(f"Error in object detection: {str(e)}", exc_info=True)
        return []

def detect_waste(image_path, model_type='cnn'):
    """
    Main function to detect waste in an image
    """
    try:
        logging.info(f"Starting waste detection for image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        logging.info(f"Image loaded successfully: {image.shape}")
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_with_boxes = image_rgb.copy()
        
        # Load model
        model = load_model_safely(model_type)
        if model is None:
            raise RuntimeError("Failed to load model")
        
        # Detect objects
        bboxes = detect_objects(image)
        
        # If no objects found, try alternative detection method
        if len(bboxes) == 0:
            logging.info("No objects found with primary method, trying alternative detection")
            # Try edge detection
            edges = cv2.Canny(image, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get image dimensions
            height, width = image.shape[:2]
            image_area = height * width
            
            # Filter contours
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area_ratio = (w * h) / image_area
                if 0.001 <= area_ratio <= 0.9:
                    bboxes.append([x, y, x + w, y + h])
            
            logging.info(f"Found {len(bboxes)} objects with alternative method")
        
        results = []
        
        # Process each detected object
        for i, bbox in enumerate(bboxes):
            try:
                x1, y1, x2, y2 = bbox
                logging.info(f"Processing object {i+1}/{len(bboxes)}")
                
                # Extract and preprocess ROI
                roi = image_rgb[y1:y2, x1:x2]
                if roi.size == 0:
                    logging.warning(f"Empty ROI for object {i+1}")
                    continue
                    
                roi_resized = cv2.resize(roi, (224, 224))
                roi_normalized = roi_resized / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)
                
                # Make prediction
                logging.info(f"Making prediction for object {i+1}")
                prediction = model.predict(roi_input, verbose=0)[0][0]
                confidence = float(prediction)
                class_idx = 0 if confidence > 0.5 else 1
                
                # Only include high confidence detections
                if confidence > 0.2:
                    class_name = "Waste" if class_idx == 0 else "No Waste"
                    color = (0, 255, 0)  # Always use green color for all detections
                    
                    # Draw box
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
                    
                    # Add to results
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_name,
                        'object_id': i + 1
                    })
                    logging.info(f"Object {i+1} classified as {class_name} with confidence {confidence:.2f}")
                    
            except Exception as e:
                logging.error(f"Error processing bbox {i}: {str(e)}", exc_info=True)
                continue
        
        logging.info(f"Detection complete. Found {len(results)} objects")
        return results, image_rgb, img_with_boxes
        
    except Exception as e:
        logging.error(f"Error in detect_waste: {str(e)}", exc_info=True)
        raise RuntimeError(f"Detection error: {str(e)}")

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    union_area = area_1 + area_2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0 