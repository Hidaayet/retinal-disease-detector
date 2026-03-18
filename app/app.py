from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import base64
import os

app = Flask(__name__)

# ── model definition ───────────────────────────────────────────────────
class RetinalClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

# ── load model ─────────────────────────────────────────────────────────
device = torch.device('cpu')
model = RetinalClassifier().to(device)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), '../data/best_model.pth'),
    map_location=device
))
model.eval()
print("Model loaded successfully")

# ── grade info ─────────────────────────────────────────────────────────
GRADES = {
    0: {
        'label': 'No DR',
        'full': 'No Diabetic Retinopathy',
        'description': 'No signs of diabetic retinopathy detected.',
        'recommendation': 'Continue regular annual eye exams.',
        'color': '#2ecc71'
    },
    1: {
        'label': 'Mild DR',
        'full': 'Mild Diabetic Retinopathy',
        'description': 'Microaneurysms only. Early stage damage.',
        'recommendation': 'Schedule follow-up in 12 months.',
        'color': '#f1c40f'
    },
    2: {
        'label': 'Moderate DR',
        'full': 'Moderate Diabetic Retinopathy',
        'description': 'More than mild but less than severe. '
                       'Vessel damage progressing.',
        'recommendation': 'Refer to ophthalmologist within 6 months.',
        'color': '#e67e22'
    },
    3: {
        'label': 'Severe DR',
        'full': 'Severe Diabetic Retinopathy',
        'description': 'Extensive damage. High risk of progression '
                       'to proliferative DR.',
        'recommendation': 'Urgent referral to ophthalmologist.',
        'color': '#e74c3c'
    },
    4: {
        'label': 'Proliferative DR',
        'full': 'Proliferative Diabetic Retinopathy',
        'description': 'Most severe stage. Abnormal new vessels '
                       'growing. High risk of vision loss.',
        'recommendation': 'Immediate ophthalmologist referral required.',
        'color': '#8e44ad'
    }
}

# ── preprocessing ──────────────────────────────────────────────────────
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = np.array(img)
    img = apply_clahe(img)
    img = Image.fromarray(img)
    return transform(img).unsqueeze(0)

# ── routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()

        # preprocess
        tensor = preprocess(image_bytes).to(device)

        # predict
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy()
            grade = int(np.argmax(probs))
            confidence = float(probs[grade])

        # build response
        grade_info = GRADES[grade]
        response = {
            'grade': grade,
            'label': grade_info['label'],
            'full': grade_info['full'],
            'description': grade_info['description'],
            'recommendation': grade_info['recommendation'],
            'color': grade_info['color'],
            'confidence': round(confidence * 100, 1),
            'all_scores': {
                GRADES[i]['label']: round(float(probs[i]) * 100, 1)
                for i in range(5)
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)