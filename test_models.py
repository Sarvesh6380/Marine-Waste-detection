import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import logging
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import signal
from datetime import datetime
from tensorflow.keras.preprocessing import image
import sys
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TimeoutError(Exception):
    pass

def timeout_handler(func, args, timeout):
    """Run a function with timeout"""
    result = [None]
    error = [None]
    
    def worker():
        try:
            result[0] = func(*args)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        thread.join(0)  # Clean up the thread
        raise TimeoutError("Operation timed out")
    
    if error[0] is not None:
        raise error[0]
        
    return result[0]

class ModelTester:
    def __init__(self, model_dir='UIFLASK/models/trained_models', test_data_dir='UIFLASK/test_data/val', batch_size=32, timeout=30):
        self.model_dir = os.path.abspath(model_dir)
        self.test_data_dir = os.path.abspath(test_data_dir)
        self.batch_size = batch_size
        self.timeout = timeout
        self.models = {}
        self.results = {}
        self.progress_file = 'test_progress.txt'
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Log initialization
        self.logger.info(f"Initializing ModelTester with:")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"Test data directory: {self.test_data_dir}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Timeout: {self.timeout} seconds")

    def load_models(self):
        """Load all pre-trained models"""
        try:
            # Load CNN model
            cnn_path = os.path.join(self.model_dir, 'cnn_model.h5')
            self.logger.info(f"Loading CNN model from {cnn_path}")
            self.models['cnn'] = load_model(cnn_path)
            self.logger.info("CNN model loaded successfully")

            # Load RCNN model
            rcnn_path = os.path.join(self.model_dir, 'rcnn_model.h5')
            self.logger.info(f"Loading RCNN model from {rcnn_path}")
            self.models['rcnn'] = load_model(rcnn_path)
            self.logger.info("RCNN model loaded successfully")

            # Load MobileNetV2 model
            mobilenet_path = os.path.join(self.model_dir, 'mobilenet_model.h5')
            self.logger.info(f"Loading MobileNetV2 model from {mobilenet_path}")
            self.models['mobilenet'] = load_model(mobilenet_path)
            self.logger.info("MobileNetV2 model loaded successfully")

            return True
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False

    def preprocess_image(self, img_path):
        """Preprocess a single image for model input"""
        try:
            self.logger.debug(f"Preprocessing image: {img_path}")
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            return img_array
        except Exception as e:
            self.logger.error(f"Error preprocessing image {img_path}: {str(e)}")
            return None

    def process_batch(self, model, image_paths):
        """Process a batch of images with timeout handling"""
        results = []
        for img_path in image_paths:
            try:
                # Preprocess image
                img_array = self.preprocess_image(img_path)
                if img_array is not None:
                    # Run prediction with timeout
                    def predict():
                        return model.predict(img_array, verbose=0)
                    
                    start_time = time.time()
                    prediction = timeout_handler(predict, (), self.timeout)
                    inference_time = time.time() - start_time
                    
                    results.append({
                        'path': img_path,
                        'prediction': prediction[0][0],
                        'inference_time': inference_time,
                        'error': None
                    })
                    self.logger.debug(f"Processed {img_path} - Prediction: {prediction[0][0]:.4f}, Time: {inference_time*1000:.2f}ms")
                
            except TimeoutError:
                self.logger.warning(f"Prediction timed out for {img_path}")
                results.append({
                    'path': img_path,
                    'prediction': None,
                    'inference_time': None,
                    'error': 'timeout'
                })
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {str(e)}")
                results.append({
                    'path': img_path,
                    'prediction': None,
                    'inference_time': None,
                    'error': str(e)
                })
            
        return results

    def evaluate_model(self, model_name, model):
        """Evaluate a single model on all test images"""
        self.logger.info(f"\nEvaluating {model_name} model...")
        
        # Get all image files
        test_images = []
        for file in os.listdir(self.test_data_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(self.test_data_dir, file)
                test_images.append(full_path)
                self.logger.debug(f"Found test image: {full_path}")
        
        if not test_images:
            self.logger.error("No test images found!")
            return None
        
        total_images = len(test_images)
        self.logger.info(f"Found {total_images} test images")
        
        # Process images in batches
        results = []
        processed = 0
        start_time = time.time()
        last_save_time = start_time
        
        for i in range(0, total_images, self.batch_size):
            batch = test_images[i:i + self.batch_size]
            batch_results = self.process_batch(model, batch)
            results.extend(batch_results)
            
            processed += len(batch)
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_image = elapsed_time / processed if processed > 0 else 0
            
            # Log progress every 10 images or 30 seconds
            if processed % 10 == 0 or (current_time - last_save_time) >= 30:
                self.logger.info(
                    f"Progress: {processed}/{total_images} images "
                    f"({processed/total_images*100:.1f}%) - "
                    f"Avg time per image: {avg_time_per_image*1000:.1f}ms - "
                    f"Elapsed: {elapsed_time:.1f}s - "
                    f"Est. remaining: {(avg_time_per_image * (total_images - processed)):.1f}s"
                )
                
                # Save intermediate results
                self._save_intermediate_results(model_name, results)
                last_save_time = current_time
        
        return results

    def _save_intermediate_results(self, model_name, results):
        """Save intermediate results to a file"""
        try:
            filename = f"intermediate_results_{model_name}.txt"
            with open(filename, 'w') as f:
                f.write(f"Intermediate results for {model_name}\n")
                f.write("=" * 50 + "\n\n")
                
                # Calculate statistics for completed predictions
                successful = [r for r in results if r['error'] is None]
                if successful:
                    avg_time = np.mean([r['inference_time'] for r in successful])
                    avg_conf = np.mean([r['prediction'] for r in successful])
                    
                    f.write(f"Processed images: {len(results)}\n")
                    f.write(f"Successful predictions: {len(successful)}\n")
                    f.write(f"Average inference time: {avg_time*1000:.2f}ms\n")
                    f.write(f"Average confidence: {avg_conf:.4f}\n\n")
                
                # List recent errors
                errors = [r for r in results if r['error'] is not None]
                if errors:
                    f.write("\nRecent errors:\n")
                    for error in errors[-5:]:  # Show last 5 errors
                        f.write(f"- {error['path']}: {error['error']}\n")
            
            self.logger.debug(f"Saved intermediate results to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {str(e)}")

    def run_tests(self):
        """Run tests on all models"""
        self.logger.info("Starting model testing...")
        
        # Check test data directory
        if not os.path.exists(self.test_data_dir):
            self.logger.error(f"Test data directory not found: {self.test_data_dir}")
            return False
            
        self.logger.info(f"Test data directory exists: {self.test_data_dir}")
        
        if not self.load_models():
            return False
        
        # Test each model
        for model_name, model in self.models.items():
            results = self.evaluate_model(model_name, model)
            if results:
                self.results[model_name] = results
        
        # Generate report
        self.generate_report()
        self.plot_performance_comparison()
        
        return True

    def generate_report(self):
        """Generate a detailed test report"""
        report_path = 'model_test_report.txt'
        with open(report_path, 'w') as f:
            f.write("Model Test Report\n")
            f.write("================\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name.upper()} Model Results:\n")
                f.write("-" * 50 + "\n")
                
                # Calculate statistics
                successful_predictions = [r for r in results if r['error'] is None]
                total_predictions = len(results)
                successful_count = len(successful_predictions)
                
                if successful_count > 0:
                    avg_inference_time = np.mean([r['inference_time'] for r in successful_predictions])
                    avg_confidence = np.mean([r['prediction'] for r in successful_predictions])
                    
                    f.write(f"Total images processed: {total_predictions}\n")
                    f.write(f"Successful predictions: {successful_count}\n")
                    f.write(f"Failed predictions: {total_predictions - successful_count}\n")
                    f.write(f"Average inference time: {avg_inference_time*1000:.2f}ms\n")
                    f.write(f"Average confidence: {avg_confidence:.4f}\n")
                else:
                    f.write("No successful predictions\n")
                
                # List errors
                errors = [r for r in results if r['error'] is not None]
                if errors:
                    f.write("\nErrors encountered:\n")
                    for error in errors:
                        f.write(f"- {error['path']}: {error['error']}\n")
                
                f.write("\n")
        
        self.logger.info(f"Report generated: {report_path}")

    def plot_performance_comparison(self):
        """Create performance comparison plots"""
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        model_names = []
        inference_times = []
        success_rates = []
        
        for model_name, results in self.results.items():
            successful = [r for r in results if r['error'] is None]
            if successful:
                avg_time = np.mean([r['inference_time'] for r in successful])
                success_rate = len(successful) / len(results) * 100
                
                model_names.append(model_name)
                inference_times.append(avg_time * 1000)  # Convert to ms
                success_rates.append(success_rate)
        
        # Plot inference times
        plt.subplot(1, 2, 1)
        plt.bar(model_names, inference_times)
        plt.title('Average Inference Time per Image')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        
        # Plot success rates
        plt.subplot(1, 2, 2)
        plt.bar(model_names, success_rates)
        plt.title('Prediction Success Rate')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png')
        self.logger.info("Performance comparison plot saved: model_performance_comparison.png")

if __name__ == "__main__":
    tester = ModelTester()
    tester.run_tests() 