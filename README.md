# DeepFake Image Detection System

A machine learning-powered web application that detects AI-generated (deepfake) images and distinguishes them from authentic photographs using Support Vector Machine (SVM) classification.

## 🎯 Overview

This project implements a full-stack solution for deepfake detection, combining machine learning models with a professional web interface. The system analyzes uploaded images and provides probability scores, confidence levels, and detailed reports to help identify whether an image is real or artificially generated.

**Key Feature**: Analyzes visual characteristics using HOG (Histogram of Oriented Gradients) features to classify images with high accuracy.

## ✨ Features

### Core Functionality
- **Image Classification**: Real vs Fake/AI-generated detection
- **Confidence Scoring**: Model prediction confidence with probability distribution
- **Risk Assessment**: Automatic risk level classification (LOW, MEDIUM, HIGH)
- **PDF Report Generation**: Professional 2-page reports with visual analytics

### Image Analysis
- **Quality Metrics**: Brightness, contrast, and sharpness analysis
- **EXIF Metadata**: Camera make/model, software, creation date extraction
- **Visual Features**: HOG-based feature extraction for classification
- **Format Support**: JPG, JPEG, PNG, WEBP (up to 10 MB)

### Security & Validation
- File type whitelist validation
- File size limits (10 MB maximum)
- Secure filename handling
- Input sanitization

## 📋 Project Structure

```
DeepFake Image Detection/
├── app.py                          # Main Flask application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/
│   ├── deepfake_svm.pkl           # Trained SVM classifier
│   └── scaler.pkl                 # StandardScaler for normalization
├── dataset/
│   ├── AI/                        # 1,001 AI-generated images
│   └── real/                      # 2,202 real images
├── templates/
│   └── index.html                 # Web interface template
├── static/
│   ├── style.css                  # Styling and layout
│   └── uploads/                   # Temporary uploaded images
└── reports/                       # Generated PDF reports
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 500 MB free disk space (for models and uploads)

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "DeepFake Image Detection"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   - Navigate to: `http://localhost:5000`
   - Or: `http://127.0.0.1:5000`

3. **Use the web interface**
   - Upload an image (JPG, JPEG, PNG, or WEBP)
   - Click "Analyze Image"
   - View results and download the PDF report

## 🔧 Technical Details

### Machine Learning Model

**Model Type**: Support Vector Machine (SVM) with RBF kernel

**Feature Extraction**: HOG (Histogram of Oriented Gradients)
- Image size: 128×128 pixels (grayscale)
- Orientations: 9
- Pixels per cell: 8×8
- Cells per block: 2×2
- Block normalization: L2-Hys

**Dataset**
- Total images: 3,203
  - Real images: 2,202 (authentic photographs)
  - AI-generated images: 1,001 (generated using text-to-image models)

**Classification**
- Class 0: REAL (Authentic)
- Class 1: FAKE (AI-generated/Deepfake)

### Risk Level Indicators

| Risk Level | Fake Probability | Interpretation |
|-----------|-----------------|-----------------|
| LOW       | < 45%           | Likely authentic image |
| MEDIUM    | 45% - 74%       | Uncertain classification |
| HIGH      | ≥ 75%           | Likely AI-generated/fake |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with upload interface |
| `/predict` | POST | Upload and analyze image |
| `/download_report/<filename>` | GET | Download PDF analysis report |
| `/health` | GET | Health check and model status |

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.1.3 | Web framework |
| OpenCV | 4.8.1.78 | Image processing |
| scikit-learn | 1.9.1 | SVM classifier |
| scikit-image | 0.26.0 | HOG feature extraction |
| NumPy | 1.26.4 | Array operations |
| Pillow | 10.4.0 | Image handling |
| ReportLab | 5.0.0 | PDF generation |
| joblib | 1.4.2 | Model serialization |
| Werkzeug | 3.1.2 | WSGI utilities |
| SciPy | 1.15.3 | Scientific computing |

## 📊 How It Works

### Image Analysis Pipeline

1. **Image Upload**
   - File validation (type, size)
   - Secure storage in temp directory

2. **Feature Extraction**
   - Convert to grayscale
   - Resize to 128×128
   - Extract HOG features

3. **Classification**
   - Scale features using pre-trained scaler
   - SVM prediction on scaled features
   - Generate probability scores

4. **Result Generation**
   - Calculate confidence level
   - Assess risk level
   - Extract image metadata
   - Analyze image quality

5. **Report Creation**
   - Generate professional PDF
   - Include visualization and metrics
   - Store for download

### Quality Metrics Explained

- **Brightness**: Average pixel intensity (0-255)
- **Contrast**: Standard deviation of pixel values
- **Sharpness**: Laplacian variance (edge detection strength)

## 🔒 Security Considerations

✅ **Implemented Security Features**
- Input file type validation
- File size restrictions (10 MB max)
- Secure filename handling
- EXIF data sanitization
- Comprehensive error handling
- Model integrity verification

⚠️ **Important Notes**
- The analysis is indicative only, not absolute proof
- Model confidence depends on training data quality
- Missing metadata doesn't indicate authenticity
- For critical applications, manual verification recommended

## 📝 Usage Examples

### Basic Image Analysis
1. Open http://localhost:5000
2. Click upload area or select image file
3. Choose a JPG, PNG, or WEBP image
4. Click "Analyze Image"
5. View results with probability scores

### Downloading Reports
1. After analysis completes
2. Click "Download PDF Report"
3. Report saved with analysis details and visualizations

## 🧪 Testing

To verify the model is working:

```python
# Health check endpoint
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true,
  "model_error": null
}
```

## 📈 Model Performance

The model achieves competitive performance on the training dataset:
- **Dataset**: 3,203 images (2,202 real + 1,001 AI-generated)
- **Feature**: HOG with configurable parameters
- **Model**: SVM with Radial Basis Function (RBF) kernel
- **Validation**: Cross-validation on training set

*Note: Actual performance metrics should be evaluated on a held-out test set*

## 🔄 Retraining the Model

To retrain the model with new data:

1. Place real images in `dataset/real/`
2. Place AI-generated images in `dataset/AI/`
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. New model files will be saved to `models/` directory

## ⚙️ Configuration

### Adjustable Parameters in `app.py`

```python
# File upload settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit

# Supported formats
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Feature extraction settings (HOG)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
```

## 🐛 Troubleshooting

### Issue: Model not loading
```
Model file not found: models/deepfake_svm.pkl
```
**Solution**: Ensure model files exist in the `models/` directory. Retrain if needed.

### Issue: Import errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: Port 5000 already in use
**Solution**: Change port in `app.py`:
```python
app.run(debug=True, port=8000)
```

### Issue: File upload fails
**Solution**: Check file format (JPG, PNG, WEBP) and size (< 10 MB)

## 📱 Browser Compatibility

- ✅ Chrome/Chromium (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Edge (latest)

## 🚢 Deployment

### For Production

1. **Disable debug mode** in `app.py`:
   ```python
   app.run(debug=False)
   ```

2. **Use production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Set up environment variables**:
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=0
   ```

4. **Use reverse proxy** (Nginx/Apache) for SSL/TLS

## 📚 Learning Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn SVM Guide](https://scikit-learn.org/stable/modules/svm.html)
- [HOG Feature Extraction](https://scikit-image.org/docs/stable/api/skimage.feature.html#hog)
- [ReportLab Documentation](https://www.reportlab.com/docs/reportlab-userguide.pdf)

## 🤝 Contributing

To contribute improvements:

1. Test your changes thoroughly
2. Ensure all dependencies are documented
3. Update README if adding new features
4. Follow PEP 8 style guidelines

## 📄 License

This project is available for educational and research purposes.

## ⚠️ Disclaimer

**Important**: This tool is designed for informational purposes and as a research project. The analysis provided is a model estimate and should not be relied upon as absolute truth. For critical applications requiring deepfake detection, consult with domain experts and use multiple verification methods.

**Ethical Considerations**:
- Use this technology responsibly
- Do not use for creating or distributing misleading content
- Respect privacy and copyright laws
- Understand the limitations of automated detection

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages in Flask console output
3. Verify all dependencies are installed correctly
4. Test with known sample images

---

**Version**: 1.0.0  
**Last Updated**: October 2026  
**Python Version**: 3.8+  
**Status**: ✅ Production Ready
