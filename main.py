from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import random

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
CLASS_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=5):
        """
        Initialize EfficientNet model with custom classifier.

        Args:
            num_classes (int): Number of classification categories
        """
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False)
        in_features = self.model.classifier.in_features

        # Custom multi-layer classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


def load_models(model_paths, num_classes, device):
    """
    Load multiple pre-trained models.

    Args:
        model_paths (list): Paths to model weights
        num_classes (int): Number of output classes
        device (torch.device): Device to load models on

    Returns:
        list: Loaded and prepared models
    """
    models = []
    for model_path in model_paths:
        model = EfficientNetModel(num_classes=num_classes).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


def preprocess_image(image_path, image_size=384):
    """
    Preprocess input image for model inference.

    Args:
        image_path (str): Path to input image
        image_size (int): Resize dimension

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms = Compose([
        Resize(image_size, image_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    transformed = transforms(image=img)
    return transformed['image'].unsqueeze(0)


def predict_with_models(models, image_tensor, device):
    """
    Perform inference with multiple models.

    Args:
        models (list): List of trained models
        image_tensor (torch.Tensor): Input image tensor
        device (torch.device): Computation device

    Returns:
        tuple: Ensemble prediction and individual model predictions
    """
    predictions = []
    all_probabilities = []
    image_tensor = image_tensor.to(device)

    for model in models:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities.max(dim=1).values.item()
            predictions.append((predicted_class, confidence))
            all_probabilities.append(probabilities.cpu().numpy()[0])

    # Ensemble prediction - average probabilities
    avg_probabilities = np.mean(all_probabilities, axis=0)
    ensemble_class = np.argmax(avg_probabilities)
    ensemble_confidence = avg_probabilities[ensemble_class]

    return (ensemble_class, ensemble_confidence), predictions, all_probabilities


def create_visualization(image_path, predictions, all_probabilities, class_labels):
    """
    Create visualization of predictions.

    Returns:
        str: Base64 encoded plot image
    """
    plt.figure(figsize=(16, 10))

    # Original Image
    plt.subplot(2, 3, 1)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('Original Fundus Image')
    plt.axis('off')

    # Ensemble Prediction
    plt.subplot(2, 3, 2)
    ensemble_probs = np.mean(all_probabilities, axis=0)
    sns.barplot(x=list(class_labels.values()), y=ensemble_probs)
    plt.title(f'Ensemble Prediction\n{class_labels[np.argmax(ensemble_probs)]}\nConfidence: {ensemble_probs.max():.2%}')
    plt.xlabel('Diabetic Retinopathy Severity')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)

    # Individual Model Predictions
    for i in range(min(3, len(predictions))):
        plt.subplot(2, 3, i + 3)
        sns.barplot(x=list(class_labels.values()), y=all_probabilities[i])
        plt.title(f'Model {i + 1}\nPrediction: {class_labels[predictions[i][0]]}\nConfidence: {predictions[i][1]:.2%}')
        plt.xlabel('Diabetic Retinopathy Severity')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)

    plt.tight_layout()

    # Convert plot to base64 string for HTML embedding
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return plot_data


# Global variables for models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = None


@app.route('/')
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    global models

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Preprocess image and make prediction
            image_tensor = preprocess_image(filepath)
            ensemble_prediction, individual_predictions, all_probabilities = predict_with_models(models, image_tensor,
                                                                                                 device)

            # Create visualization
            plot_data = create_visualization(filepath, individual_predictions, all_probabilities, CLASS_LABELS)

            # Prepare results
            result = {
                'filename': file.filename,
                'filepath': filepath,
                'ensemble_prediction': {
                    'class_id': int(ensemble_prediction[0]),
                    'class_name': CLASS_LABELS[ensemble_prediction[0]],
                    'confidence': float(ensemble_prediction[1])
                },
                'individual_predictions': [
                    {
                        'model_id': i + 1,
                        'class_id': int(pred[0]),
                        'class_name': CLASS_LABELS[pred[0]],
                        'confidence': float(pred[1])
                    } for i, pred in enumerate(individual_predictions)
                ],
                'plot': plot_data
            }

            return render_template('result.html', result=result)

        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'Invalid file type'})


@app.route('/about')
def about():
    return render_template('about.html')


def initialize_models():
    global models

    set_seed(42)
    model_paths = [
        'models/best_model_fold_1 (1).pth',
        'models/best_model_fold_2 (1).pth',
        'models/best_model_fold_3 (1).pth',
        'models/best_model_fold_4 (1).pth',
        'models/best_model_fold_5 (1).pth'
    ]

    # Check if model files exist
    missing_models = [path for path in model_paths if not os.path.exists(path)]
    if missing_models:
        print(f"Warning: The following model files are missing: {missing_models}")
        print("Please ensure all model files are in the correct location.")

    models = load_models(model_paths, num_classes=len(CLASS_LABELS), device=device)
    print(f"Models loaded successfully. Using device: {device}")


if __name__ == '__main__':
    # Initialize models at startup
    initialize_models()
    app.run(debug=True)