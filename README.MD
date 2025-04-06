# Diabetic Retinopathy Detection Web Application

![Diabetic Retinopathy Detection](https://img.shields.io/badge/Medical%20AI-Diabetic%20Retinopathy%20Detection-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-orange)

A web-based application for automated detection and classification of Diabetic Retinopathy severity levels from fundus images. This project utilizes deep learning models trained on the APTOS 2019 Blindness Detection dataset to provide accessible eye disease screening.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🔍 Overview

Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes and is one of the leading causes of blindness globally. Early detection is critical for preventing vision loss. This application provides an accessible tool for screening fundus images for signs of DR and classifying them into appropriate severity levels.

The web application leverages state-of-the-art deep learning models to analyze retinal images and provide instant classification results, making DR screening more accessible to healthcare providers, especially in areas with limited access to ophthalmologists.

## ✨ Features

- **Web-based Interface**: Easy-to-use web interface for image upload and analysis
- **Ensemble Predictions**: Combines predictions from 5 different models for more robust results
- **Visualization**: Visual representation of model predictions and confidence scores
- **Educational Content**: Information about DR and what different severity levels mean
- **Comprehensive Results**: Detailed analysis including individual model predictions
- **Responsive Design**: Works on desktop and mobile devices

## 📊 Dataset

This project uses models trained on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset from Kaggle. The dataset consists of retinal images taken using fundus photography under varying imaging conditions.

The dataset contains images with the following DR severity levels:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

## 🧠 Models

The application uses an ensemble of 5 EfficientNet-B0 models trained via k-fold cross-validation. These models were trained by [Anett Varghese on Kaggle](https://www.kaggle.com/models/anettvarghese/blindness_efficentnetb0) and achieve high accuracy on the APTOS dataset.

Model architecture:
- Base: EfficientNet-B0
- Custom multilayer classifier with dropout for better generalization
- Ensemble approach combining 5 models for improved reliability

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models:
   - Download the models from [Kaggle](https://www.kaggle.com/models/anettvarghese/blindness_efficentnetb0)
   - Place the model files in the `models/` directory:
     - `models/best_model_fold_1 (1).pth`
     - `models/best_model_fold_2 (1).pth`
     - `models/best_model_fold_3 (1).pth`
     - `models/best_model_fold_4 (1).pth`
     - `models/best_model_fold_5 (1).pth`

## 💻 Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Upload a fundus image through the web interface.

4. View the analysis results showing DR severity classification, confidence scores, and visualizations.

## 📁 Project Structure

```
diabetic-retinopathy-detection/
├── main.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models/                # Trained model files
│   ├── models/best_model_fold_1 (1).pth
│   ├── models/best_model_fold_2 (1).pth
│   ├── models/best_model_fold_3 (1).pth
│   ├── models/best_model_fold_4 (1).pth
│   └── models/best_model_fold_5 (1).pth
├── static/                # Static files
│   └── uploads/           # For uploaded images
├── templates/             # HTML templates
│   ├── index.html         # Homepage
│   ├── result.html        # Results page
│   └── about.html         # Information page
└── README.md              # Project documentation
```


## 🔮 Future Enhancements

- User accounts for tracking patient history
- PDF report generation for healthcare providers
- API endpoints for mobile application integration
- Additional metrics and measurements
- Database integration for result storage
- Docker containerization for easier deployment
- Telemedicine integration capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset
- [Anett Varghese](https://www.kaggle.com/models/anettvarghese/blindness_efficentnetb0) for the pre-trained models
- [SrisairamGautamB](https://www.kaggle.com/code/srisairamgautamb/blindness-realtime) for the base code inspiration
- [Timm library](https://github.com/rwightman/pytorch-image-models) for EfficientNet implementation
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the frontend design

---

⚠️ **Medical Disclaimer**: This tool is for educational and screening purposes only. It is not intended to replace professional medical diagnosis. Always consult with healthcare professionals for proper diagnosis and treatment of diabetic retinopathy and other eye conditions.