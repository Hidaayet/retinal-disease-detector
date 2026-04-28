import io
import logging
import os
import base64

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
LOW_CONFIDENCE_THRESHOLD = 0.60

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


# ── Grad-CAM ───────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target = model.backbone.conv_head
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(300, 300), mode="bilinear", align_corners=False)
        cam = cam.squeeze().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def _overlay_heatmap(image_bytes: bytes, cam: np.ndarray) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((300, 300))
    img_np = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_np + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── load model ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
model = None
grad_cam = None

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "data", "best_model.pth")

try:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}.")

    model = RetinalClassifier().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    grad_cam = GradCAM(model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("✓ Model loaded — %d parameters", n_params)

except FileNotFoundError as exc:
    logger.error("✗ %s", exc)
except Exception:
    logger.exception("✗ Failed to load model")


# ── grade metadata ─────────────────────────────────────────────────────────
GRADES = {
    0: {
        "label": "No DR",
        "full": "No Diabetic Retinopathy",
        "description": "No signs of diabetic retinopathy detected in this image.",
        "recommendation": "Continue routine annual diabetic eye screening. Maintain glycemic and blood pressure control.",
        "color": "#00875a",
    },
    1: {
        "label": "Mild NPDR",
        "full": "Mild Non-Proliferative Diabetic Retinopathy",
        "description": "Microaneurysms only — earliest detectable sign of DR.",
        "recommendation": "Schedule follow-up within 6 months. Reinforce glycemic control.",
        "color": "#0066cc",
    },
    2: {
        "label": "Moderate NPDR",
        "full": "Moderate Non-Proliferative Diabetic Retinopathy",
        "description": "More than mild but less than severe NPDR. Hemorrhages and/or exudates present.",
        "recommendation": "Refer to ophthalmology within 1–2 months. Consider panretinal photocoagulation if indicated.",
        "color": "#b45309",
    },
    3: {
        "label": "Severe NPDR",
        "full": "Severe Non-Proliferative Diabetic Retinopathy",
        "description": "Extensive hemorrhages, venous beading, or intraretinal microvascular abnormalities.",
        "recommendation": "Urgent ophthalmology referral within 1 week. High risk of progression to PDR.",
        "color": "#c0392b",
    },
    4: {
        "label": "Proliferative DR",
        "full": "Proliferative Diabetic Retinopathy",
        "description": "Neovascularization present. Risk of vitreous hemorrhage and tractional detachment.",
        "recommendation": "Emergency ophthalmology referral. Anti-VEGF or panretinal laser treatment required.",
        "color": "#6d28d9",
    },
}


# ── preprocessing ──────────────────────────────────────────────────────────
def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = _apply_clahe(np.array(img))
    return _transform(Image.fromarray(img_np)).unsqueeze(0)


# ── routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "grad_cam_ready": grad_cam is not None,
        "model_path": MODEL_PATH,
        "model_file_exists": os.path.isfile(MODEL_PATH),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not available. Visit /health to diagnose."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload PNG or JPG."}), 400

    image_bytes = file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return jsonify({"error": f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB."}), 400

    try:
        tensor = _preprocess(image_bytes).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy()

        grade = int(np.argmax(probs))
        confidence = float(probs[grade])
        grade_info = GRADES[grade]
        low_conf = confidence < LOW_CONFIDENCE_THRESHOLD

        logger.info("Prediction: grade=%d (%s)  confidence=%.1f%%", grade, grade_info["label"], confidence * 100)

        cam_image_b64 = None
        if grad_cam is not None:
            try:
                tensor_grad = _preprocess(image_bytes).to(device)
                cam = grad_cam.generate(tensor_grad, grade)
                cam_image_b64 = _overlay_heatmap(image_bytes, cam)
            except Exception:
                logger.warning("Grad-CAM generation failed — skipping heatmap", exc_info=True)

        return jsonify({
            "grade": grade,
            "label": grade_info["label"],
            "full": grade_info["full"],
            "description": grade_info["description"],
            "recommendation": grade_info["recommendation"],
            "color": grade_info["color"],
            "confidence": round(confidence * 100, 1),
            "probabilities": [round(float(probs[i]), 4) for i in range(5)],
            "all_scores": {GRADES[i]["label"]: round(float(probs[i]) * 100, 1) for i in range(5)},
            "cam_image": cam_image_b64,
            "low_confidence": low_conf,
            "low_confidence_warning": (
                "⚠ Low confidence — result may be unreliable. Please consult a qualified ophthalmologist."
                if low_conf else None
            ),
        })

    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Analysis failed. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)