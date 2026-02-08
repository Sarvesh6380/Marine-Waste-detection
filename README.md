# Marine Waste Detection System

A web-based application for detecting and classifying marine waste using deep learning models (CNN, RCNN, MobileNetV2).

---

## Features
- Upload images and detect waste using multiple AI models
- Visualize detection results and analytics
- Download full project documentation (PDF)
- User-friendly web interface

---

## Prerequisites
- Python 3.8+(USE Python 3.11.7)
- pip (Python package manager)

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd UIFLASK
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Trained Models are Present**
   - The following files should be in `models/trained_models/`:
     - `cnn_model.h5`
     - `rcnn_model.h5`
     - `mobilenet_model.h5`
     - `waste_detection_model.h5`
   - If not present, place your trained model files in this directory.

5. **Ensure the `uploads/` directory exists**
   - This is used for temporary image uploads. It should be created automatically, but you can create it manually if needed.

6. **Run the Flask app**
   ```bash
   python app.py
   ```

7. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

---

## Usage
- **Home Page:** Upload an image, select a model, and click "Detect Waste" to see results.
- **Abstract:** Read the project abstract.
- **Model Info:** View details about each AI model.
- **Contact:** Contact the team for support or collaboration.
- **Documentation:** Download the full project documentation from the footer (PDF).

---

## Project Structure
```
UIFLASK/
├── app.py
├── requirements.txt
├── models/
│   ├── detect_waste.py
│   └── trained_models/
│       ├── cnn_model.h5
│       ├── rcnn_model.h5
│       ├── mobilenet_model.h5
│       └── waste_detection_model.h5
├── templates/
│   ├── index.html
│   ├── abstract.html
│   ├── contact.html
│   └── model_info.html
├── static/
│   ├── docs/
│   │   └── BLACKBOOK_FINAL_DRAFTT.pdf
│   ├── images/
│   └── uploads/
├── uploads/
├── data/ (optional, for training/testing)
├── test_data/ (optional, for testing)
├── dataset/ (optional, for training/testing)
└── ...
```

---

## Notes
- For best performance, use a machine with a compatible GPU for TensorFlow.
- If you want to retrain models, use the scripts and data directories as needed.
- For any issues, contact the team via the Contact page.

---

## License
This project is for academic and research purposes. 