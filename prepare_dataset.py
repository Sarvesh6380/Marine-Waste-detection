import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img = img / 255.0
    
    return img

def process_directory(source_dir, target_dir):
    """Process all images in a directory and save as numpy arrays"""
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Process each image in the source directory
    for filename in os.listdir(source_dir):
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            continue
            
        source_path = os.path.join(source_dir, filename)
        
        # Preprocess and save the image
        processed_img = preprocess_image(source_path)
        if processed_img is not None:
            # Save as numpy array for faster loading during training
            save_path = os.path.join(target_dir, os.path.splitext(filename)[0] + '.npy')
            np.save(save_path, processed_img)
            
            # Also save as image for visualization
            cv2.imwrite(
                os.path.join(target_dir, filename),
                cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            )

def main():
    """Main function to prepare the dataset"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Prepare dataset for waste detection models')
    parser.add_argument('--base_dir', type=str, required=True,
                      help='Path to the base directory containing train, test, and valid folders')
    args = parser.parse_args()
    
    # Define paths
    base_dir = args.base_dir
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    valid_dir = os.path.join(base_dir, 'valid')
    
    # Create processed dataset directory
    processed_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process each directory
    print("Processing training data...")
    process_directory(train_dir, os.path.join(processed_dir, 'train'))
    
    print("Processing test data...")
    process_directory(test_dir, os.path.join(processed_dir, 'test'))
    
    print("Processing validation data...")
    process_directory(valid_dir, os.path.join(processed_dir, 'valid'))
    
    print("Dataset preparation completed successfully!")

if __name__ == '__main__':
    main() 