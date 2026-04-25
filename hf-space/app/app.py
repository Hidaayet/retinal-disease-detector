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
# MUST match the architecture used during training exactly.
# EfficientNet-B3 backbone → Dropout → Linear(1536→256) → ReLU → Dropout → Linear(256→5)
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
    """Gradient-weighted Class Activation Mapping for EfficientNet-B3.

    Hooks onto conv_head (the final conv layer before global pooling).
    NOTE: generate() temporarily enables grad mode even when the model is in
    eval() — this is intentional and does NOT affect batch-norm statistics.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # conv_head is the last conv before the classifier — best Grad-CAM target
        target = model.backbone.conv_head
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # Store activations (detached so they won't keep the graph alive)
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a normalised (0–1) CAM array of shape (300, 300)."""
        # Grad-CAM requires gradients — temporarily enable them even in eval mode
        self.model.zero_grad()

        # Run forward WITHOUT no_grad so the graph is built for backward()
        output = self.model(input_tensor)

        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot)

        # Both hooks must have fired by now
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire — check hook target layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,H,W)
        cam = F.interpolate(cam, size=(300, 300), mode="bilinear", align_corners=False)
        cam = cam.squeeze().numpy()  # (300, 300)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def _overlay_heatmap(image_bytes: bytes, cam: np.ndarray) -> str:
    """Blend Grad-CAM heatmap onto the original image; return a base64 PNG data-URI."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((300, 300))
    img_np = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_np + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── load model ─────────────────────────────────────────────────────────────
# The model file is expected at  <repo_root>/data/best_model.pth
# Inside Docker this resolves to /app/data/best_model.pth  (Dockerfile sets WORKDIR /app
# and does: COPY data/best_model.pth ./data/best_model.pth)
device = torch.device("cpu")
model: RetinalClassifier | None = None
grad_cam: GradCAM | None = None

# ── FIX: resolve model path robustly ──────────────────────────────────────
# __file__ = /app/app/app.py  →  dirname = /app/app  →  ../data = /app/data ✓
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "data", "best_model.pth")

try:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            "Make sure best_model.pth is committed to hf-space/data/ via Git LFS."
        )

    model = RetinalClassifier().to(device)
    # ── FIX: train.py saves torch.save(model.state_dict(), path) — a raw OrderedDict,
    # NOT wrapped in {"model_state_dict": ...}.  We handle all three formats defensively.
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Raw state_dict saved directly (current train.py behaviour)
            state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    # strict=True will raise immediately if there is any key mismatch,
    # making silent "random-weights" failures impossible.
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    grad_cam = GradCAM(model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("✓ Model loaded — %d parameters — path: %s", n_params, MODEL_PATH)

except FileNotFoundError as exc:
    logger.error("✗ %s", exc)
except Exception:
    logger.exception("✗ Failed to load model — predictions unavailable")


# ── grade metadata ─────────────────────────────────────────────────────────
GRADES = {
    0: {
        "label": "No DR",
        "full": "No Diabetic Retinopathy",
        "description": "No signs of diabetic retinopathy detected in this image.",
        "recommendation": (
            "Continue routine annual diabetic eye screening. "
            "Maintain glycemic and blood pressure control."
        ),
        "color": "#00875a",
    },
    1: {
        "label": "Mild NPDR",
        "full": "Mild Non-Proliferative Diabetic Retinopathy",
        "description": "Microaneurysms only — earliest detectable sign of DR.",
        "recommendation": (
            "Schedule follow-up within 6 months. Reinforce glycemic control."
        ),
        "color": "#0066cc",
    },
    2: {
        "label": "Moderate NPDR",
        "full": "Moderate Non-Proliferative Diabetic Retinopathy",
        "description": (
            "More than mild but less than severe NPDR. "
            "Hemorrhages and/or exudates present."
        ),
        "recommendation": (
            "Refer to ophthalmology within 1–2 months. "
            "Consider panretinal photocoagulation if indicated."
        ),
        "color": "#b45309",
    },
    3: {
        "label": "Severe NPDR",
        "full": "Severe Non-Proliferative Diabetic Retinopathy",
        "description": (
            "Extensive hemorrhages, venous beading, or "
            "intraretinal microvascular abnormalities."
        ),
        "recommendation": (
            "Urgent ophthalmology referral within 1 week. "
            "High risk of progression to PDR."
        ),
        "color": "#c0392b",
    },
    4: {
        "label": "Proliferative DR",
        "full": "Proliferative Diabetic Retinopathy",
        "description": (
            "Neovascularization present. "
            "Risk of vitreous hemorrhage and tractional detachment."
        ),
        "recommendation": (
            "Emergency ophthalmology referral. "
            "Anti-VEGF or panretinal laser treatment required."
        ),
        "color": "#6d28d9",
    },
}

# ── preprocessing ──────────────────────────────────────────────────────────
# Must exactly match the pipeline in model/dataset.py used during training:
#   1. Load image as RGB
#   2. Apply CLAHE on the L channel (LAB colour space)
#   3. Resize to 300×300
#   4. ToTensor + ImageNet normalisation
def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of an LAB image (matches training dataset.py)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


# Validation-time transform (no augmentation) — mirrors get_transforms(is_train=False)
_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    """Read raw image bytes → normalised (1, 3, 300, 300) tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = _apply_clahe(np.array(img))
    return _transform(Image.fromarray(img_np)).unsqueeze(0)


# ── routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Liveness check — call this to verify the model loaded correctly."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "grad_cam_ready": grad_cam is not None,
        "model_path": MODEL_PATH,
        "model_file_exists": os.path.isfile(MODEL_PATH),
    })


@app.route("/debug", methods=["POST"])
def debug():
    """Debug endpoint — returns raw logits and softmax probs for any uploaded image.

    Usage:
        curl -X POST https://<space>/debug -F "file=@image.jpg" | python -m json.tool
    """
    if model is None:
        return jsonify({"error": "model not loaded — check /health for details"}), 503
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    try:
        tensor = _preprocess(image_bytes).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy()

        predicted = int(np.argmax(probs))
        return jsonify({
            "raw_logits":      outputs[0].numpy().tolist(),
            "probabilities":   probs.tolist(),
            "predicted_grade": predicted,
            "predicted_label": GRADES[predicted]["label"],
            "tensor_shape":    list(tensor.shape),
            "tensor_mean":     round(float(tensor.mean()), 6),
            "tensor_std":      round(float(tensor.std()), 6),
            "model_path":      MODEL_PATH,
        })
    except Exception as e:
        logger.exception("Debug endpoint error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    # ── guard: model must be loaded ────────────────────────────────────────
    if model is None:
        return jsonify({
            "error": (
                "Model not available. "
                "Visit /health to diagnose — the weights file may be missing."
            )
        }), 503

    # ── guard: file presence & type ────────────────────────────────────────
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
        return jsonify({
            "error": f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB."
        }), 400

    try:
        # ── inference (no_grad for memory efficiency) ──────────────────────
        tensor = _preprocess(image_bytes).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0].numpy()

        grade      = int(np.argmax(probs))
        confidence = float(probs[grade])
        grade_info = GRADES[grade]
        low_conf   = confidence < LOW_CONFIDENCE_THRESHOLD

        logger.info(
            "Prediction: grade=%d (%s)  confidence=%.1f%%",
            grade, grade_info["label"], confidence * 100,
        )

        # ── Grad-CAM (separate forward with gradients enabled) ─────────────
        # We intentionally call _preprocess again so the inference tensor
        # (computed under no_grad) is not entangled with the grad-CAM graph.
        cam_image_b64 = None
        if grad_cam is not None:
            try:
                # grad_cam.generate() calls backward() internally — must NOT
                # be inside a torch.no_grad() block.
                tensor_grad = _preprocess(image_bytes).to(device)
                cam = grad_cam.generate(tensor_grad, grade)
                cam_image_b64 = _overlay_heatmap(image_bytes, cam)
                logger.info("Grad-CAM generated successfully")
            except Exception:
                logger.warning("Grad-CAM generation failed — skipping heatmap", exc_info=True)

        # ── build response ─────────────────────────────────────────────────
        return jsonify({
            "grade":          grade,
            "label":          grade_info["label"],
            "full":           grade_info["full"],
            "description":    grade_info["description"],
            "recommendation": grade_info["recommendation"],
            "color":          grade_info["color"],
            "confidence":     round(confidence * 100, 1),
            # Per-class softmax probabilities (raw, sum to 1)
            "probabilities":  [round(float(probs[i]), 4) for i in range(5)],
            # Human-readable percentage per class for the UI bars
            "all_scores":     {
                GRADES[i]["label"]: round(float(probs[i]) * 100, 1)
                for i in range(5)
            },
            "cam_image":      cam_image_b64,
            "low_confidence": low_conf,
            "low_confidence_warning": (
                "⚠ Low confidence — result may be unreliable. "
                "Please consult a qualified ophthalmologist."
                if low_conf else None
            ),
        })

    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Analysis failed. Please try again."}), 500


if __name__ == "__main__":
    # Development only — production uses Gunicorn (see Dockerfile CMD)
    app.run(host="0.0.0.0", port=7860, debug=False)