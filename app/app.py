import io
import logging
import os

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── constants ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
LOW_CONFIDENCE_THRESHOLD = 0.60   # warn user below this

# ── model definition ───────────────────────────────────────────────────────
class RetinalClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ── load model ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
model = None

try:
    model = RetinalClassifier().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "../data/best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully from %s", model_path)
except Exception:
    logger.exception("Failed to load model — predictions will be unavailable")

# ── grade info ─────────────────────────────────────────────────────────────
GRADES = {
    0: {
        "label": "No DR",
        "full": "No Diabetic Retinopathy",
        "description": "No signs of diabetic retinopathy detected.",
        "recommendation": "Continue regular annual eye exams.",
        "color": "#2ecc71",
    },
    1: {
        "label": "Mild DR",
        "full": "Mild Diabetic Retinopathy",
        "description": "Microaneurysms only. Early stage damage.",
        "recommendation": "Schedule follow-up in 12 months.",
        "color": "#f1c40f",
    },
    2: {
        "label": "Moderate DR",
        "full": "Moderate Diabetic Retinopathy",
        "description": "More than mild but less than severe. Vessel damage progressing.",
        "recommendation": "Refer to ophthalmologist within 6 months.",
        "color": "#e67e22",
    },
    3: {
        "label": "Severe DR",
        "full": "Severe Diabetic Retinopathy",
        "description": "Extensive damage. High risk of progression to proliferative DR.",
        "recommendation": "Urgent referral to ophthalmologist.",
        "color": "#e74c3c",
    },
    4: {
        "label": "Proliferative DR",
        "full": "Proliferative Diabetic Retinopathy",
        "description": "Most severe stage. Abnormal new vessels growing. High risk of vision loss.",
        "recommendation": "Immediate ophthalmologist referral required.",
        "color": "#8e44ad",
    },
}

# ── helpers ────────────────────────────────────────────────────────────────
def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _apply_clahe(np.array(img))
    return _transform(Image.fromarray(img)).unsqueeze(0)


# ── routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ── model availability ──────────────────────────────────────────────────
    if model is None:
        return jsonify({"error": "Model is not available. Please try again later."}), 503

    # ── file presence ───────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # ── file type ───────────────────────────────────────────────────────────
    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Please upload a PNG or JPG image."
        }), 400

    # ── file size ───────────────────────────────────────────────────────────
    image_bytes = file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return jsonify({
            "error": f"File too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        }), 400

    # ── inference ───────────────────────────────────────────────────────────
    try:
        tensor = _preprocess(image_bytes).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy()
            grade = int(np.argmax(probs))
            confidence = float(probs[grade])

        low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
        grade_info = GRADES[grade]

        response = {
            "grade": grade,
            "label": grade_info["label"],
            "full": grade_info["full"],
            "description": grade_info["description"],
            "recommendation": grade_info["recommendation"],
            "color": grade_info["color"],
            "confidence": round(confidence * 100, 1),
            "low_confidence": low_confidence,
            "low_confidence_warning": (
                "⚠️ Confidence is low — this result may be unreliable. "
                "Please consult a qualified ophthalmologist."
                if low_confidence else None
            ),
            "all_scores": {
                GRADES[i]["label"]: round(float(probs[i]) * 100, 1)
                for i in range(5)
            },
        }
        logger.info(
            "Prediction: grade=%d confidence=%.2f low_conf=%s",
            grade, confidence, low_confidence,
        )
        return jsonify(response)

    except Exception:
        logger.exception("Prediction failed for uploaded file")
        return jsonify({"error": "An error occurred during analysis. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)