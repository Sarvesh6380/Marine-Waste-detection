import numpy as np
import cv2
import os

def create_sample_images(output_dir='test_data', num_images=5):
    """Create sample images for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create different types of sample images
    for i in range(num_images):
        # Create a random colored image
        img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
        # Add some shapes to simulate waste objects
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.circle(img, (300, 300), 50, (0, 0, 255), -1)
        
        # Save the image
        filename = os.path.join(output_dir, f'test_image_{i+1}.jpg')
        cv2.imwrite(filename, img)
        print(f"Created {filename}")

if __name__ == "__main__":
    create_sample_images() 