import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from models.detect_waste import detect_waste
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models/trained_models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/model_info')
def model_info():
    model_info = {
        'cnn': {
            'name': 'Convolutional Neural Network',
            'description': 'High-accuracy deep learning model for waste detection',
            'accuracy': '96%',
            'speed': 'Fast (45ms)',
            'memory': '85MB'
        },
        'rcnn': {
            'name': 'Region-based CNN',
            'description': 'Advanced region-based detection model',
            'accuracy': '92%',
            'speed': 'Medium (75ms)',
            'memory': '120MB'
        },
        'mobilenet': {
            'name': 'MobileNetV2',
            'description': 'Lightweight model for mobile devices',
            'accuracy': '88%',
            'speed': 'Very Fast (25ms)',
            'memory': '45MB'
        }
    }
    return render_template('model_info.html', model_info=model_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            logging.error('No file part in request')
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logging.error('No selected file')
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logging.error(f'Invalid file type: {file.filename}')
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Get model type
        model_type = request.form.get('model_type', 'cnn')
        logging.info(f'Processing file with model type: {model_type}')
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f'File saved to: {filepath}')
        
        # Process the image
        try:
            # Get results from detect_waste
            results, image_rgb, img_with_boxes = detect_waste(filepath, model_type)
            logging.info('Detection completed successfully')
            
            # Convert images to base64
            _, buffer_original = cv2.imencode('.png', image_rgb)
            original_image = base64.b64encode(buffer_original).decode('utf-8')
            
            _, buffer_boxes = cv2.imencode('.png', img_with_boxes)
            result_image = base64.b64encode(buffer_boxes).decode('utf-8')
            
            # Create comparison plot of actual vs predicted values
            plt.figure(figsize=(14, 9))
            if results:
                # Extract actual and predicted values
                # For this example, we'll use the class as actual and confidence as predicted
                # In a real application, you would have ground truth data
                actual_values = [1 if r['class'] == 'Waste' else 0 for r in results]
                predicted_values = [r['confidence'] for r in results]
                
                # Create bar chart comparing actual vs predicted
                x = np.arange(len(results))
                width = 0.35
                
                # Create the bars with better colors and transparency
                plt.bar(x - width/2, actual_values, width, label='Actual', color='#3498db', alpha=0.8)
                plt.bar(x + width/2, predicted_values, width, label='Predicted', color='#e74c3c', alpha=0.8)
                
                # Add value labels on top of bars
                for i, v in enumerate(actual_values):
                    plt.text(i - width/2, v + 0.05, f'{v:.1f}', ha='center', va='bottom', fontsize=12)
                for i, v in enumerate(predicted_values):
                    plt.text(i + width/2, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
                
                # Customize the plot
                plt.title('Object Distribution: Actual vs Predicted Values', fontsize=24, pad=20, color='#2c3e50')
                plt.xlabel('Object ID', fontsize=14)
                plt.ylabel('Value (1=Waste, 0=No Waste)', fontsize=14)
                plt.xticks(x, [f"Obj {r['object_id']}" for r in results], fontsize=12)
                plt.yticks([0, 0.25, 0.5, 0.75, 1.0], fontsize=12)
                plt.legend(loc='upper right', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.3, axis='y')
                
                # Add a horizontal line at y=0.5 to indicate the threshold
                plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                plt.text(len(results)-0.5, 0.52, 'Threshold', fontsize=12, color='gray')
                
                # Ensure the plot is properly sized and positioned
                plt.tight_layout(pad=3.0)
                
                # Save the plot with high DPI for better quality
                plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'comparison_plot.png')
                plt.savefig(plot_path, bbox_inches='tight', dpi=150, pad_inches=0.3)
                plt.close('all')
                
                # Read the plot as base64
                with open(plot_path, 'rb') as f:
                    plot_data = base64.b64encode(f.read()).decode('utf-8')
                os.remove(plot_path)
            else:
                plot_data = None
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'original_image': original_image,
                'image': result_image,
                'plot': plot_data,
                'total_objects': len(results),
                'waste_count': sum(1 for r in results if r['class'] == 'Waste'),
                'no_waste_count': sum(1 for r in results if r['class'] == 'No Waste'),
                'results': results
            })
            
        except Exception as e:
            logging.error(f'Error during detection: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
            
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

