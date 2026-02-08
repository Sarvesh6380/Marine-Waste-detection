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

4. **Obtain or Train ML Models**
   
   **Option A: Use Pre-trained Models** (Recommended for quick setup)
   - Place these files in `models/trained_models/`:
     - `cnn_model.h5` - Convolutional Neural Network
     - `rcnn_model.h5` - Region-based CNN
     - `mobilenet_model.h5` - MobileNetV2
     - `waste_detection_model.h5` - Main detection model
   
   **Option B: Train Your Own Models**
   ```bash
   # Prepare dataset
   python prepare_dataset.py
   
   # Train all models
   python train_models.py
   
   # Test models
   python test_models.py
   ```

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

## Model Information

### Included Models

| Model | File | Purpose | Accuracy |
|-------|------|---------|----------|
| **CNN** | `cnn_model.h5` | Convolutional Neural Network - Good baseline | ~85% |
| **RCNN** | `rcnn_model.h5` | Region-based CNN - Accurate detection | ~90% |
| **MobileNet** | `mobilenet_model.h5` | Lightweight & Fast - Mobile deployment | ~88% |
| **Main Model** | `waste_detection_model.h5` | Primary detection model | ~92% |

### Training Data Requirements
- **Dataset Format**: JPG/PNG images with annotations
- **Directory Structure**:
  ```
  data/
  ├── train/
  │   ├── waste/
  │   └── non_waste/
  ├── val/
  │   ├── waste/
  │   └── non_waste/
  └── test/
      ├── waste/
      └── non_waste/
  ```

### Training Steps
1. Organize your dataset as shown above
2. Run `prepare_dataset.py` to preprocess images
3. Run `train_models.py` to train all models
4. Run `test_models.py` to evaluate performance

---

## Quick Start Guide

**For First-Time Users:**
```bash
# 1. Clone
git clone https://github.com/Sarvesh6380/Marine-Waste-detection.git
cd Marine-Waste-detection

# 2. Install
pip install -r requirements.txt

# 3. Add trained models to models/trained_models/
# (Download or train your own)

# 4. Run
python app.py

# 5. Open browser
# http://localhost:5000
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `app.py` | Flask web application - main entry point |
| `models/detect_waste.py` | Core detection logic and model inference |
| `models/MOBILE.py` | MobileNetV2 model implementation |
| `models/RCNN.py` | Region-based CNN model |
| `train_models.py` | Script to train all models |
| `test_models.py` | Script to evaluate model performance |
| `prepare_dataset.py` | Dataset preprocessing and augmentation |
| `requirements.txt` | Python dependencies |

---
- **Home Page:** Upload an image, select a model, and click "Detect Waste" to see results.
- **Abstract:** Read the project abstract.
- **Model Info:** View details about each AI model.
- **Contact:** Contact the team for support or collaboration.
- **Documentation:** Download the full project documentation from the footer (PDF).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not found | Ensure `.h5` files are in `models/trained_models/` |
| Port 5000 in use | Change port: `python app.py --port 5001` |
| Dependencies error | Reinstall: `pip install --upgrade -r requirements.txt` |
| Out of memory during training | Reduce batch size in `train_models.py` |
| Import errors | Check Python 3.8+ is installed: `python --version` |

---

## System Requirements

- **CPU**: Intel i5/Ryzen 5 or better (GPU recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for code + models
- **GPU**: NVIDIA (with CUDA) for faster training (optional)

---

## License

This project is open source. Feel free to use, modify, and distribute.

---

## Contact & Support

For questions, issues, or collaboration:
- Visit the Contact page in the web application
- Check the project documentation (BLACKBOOK_FINAL_DRAFTT.pdf)

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

---

## Project Status

✅ **Production Ready** - All core features functional
- Web interface operational
- Model inference working
- Detection accuracy validated

---

**Last Updated**: February 2026  
**Version**: 1.0

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