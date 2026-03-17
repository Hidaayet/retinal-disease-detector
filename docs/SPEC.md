# Project Specification — Retinal Disease Detector

**Author:** Hidayet Allah Yaakoubi  
**Date:** March 2026  
**Status:** In progress  

---

## 1. Project summary

A deep learning system that classifies diabetic retinopathy severity from
fundus retinal photographs. A fine-tuned EfficientNet-B3 model processes
uploaded images and returns a severity grade (0–4) with confidence scores.
The system is deployed as a Flask web application accessible via browser.

---

## 2. Goals

- Fine-tune EfficientNet-B3 on APTOS 2019 retinal dataset
- Achieve above 80% accuracy on the validation set
- Deploy as a working web application
- Provide confidence scores and grade explanation per prediction
- Produce clean, documented, reproducible code

---

## 3. Dataset — APTOS 2019 Blindness Detection

| Property | Value |
|---|---|
| Source | Kaggle — APTOS 2019 Blindness Detection |
| URL | https://www.kaggle.com/competitions/aptos2019-blindness-detection |
| Training images | 3,662 fundus photographs |
| Image format | PNG, variable resolution |
| Labels | 0–4 severity grade per image |
| Task | 5-class classification |

**Class distribution:**

| Grade | Label | Count | % |
|---|---|---|---|
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative | 295 | 8.1% |

Note: dataset is imbalanced — grade 0 dominates. Handled via weighted loss.

---

## 4. Model — EfficientNet-B3

**Why EfficientNet-B3:**
- State-of-the-art accuracy vs parameter count tradeoff
- Pretrained on ImageNet — strong visual feature extractor
- B3 variant balances accuracy and speed for medical imaging
- Widely used in published retinal disease detection research

**Architecture modification:**
- Remove final classification head
- Add dropout (0.3) + new linear layer → 5 outputs
- Fine-tune entire network with low learning rate

**Training strategy:**
- Phase 1: freeze backbone, train head only (5 epochs)
- Phase 2: unfreeze all layers, fine-tune end-to-end (15 epochs)
- Optimizer: Adam, lr=0.0001
- Loss: CrossEntropyLoss with class weights
- Augmentation: random flip, rotation, color jitter, zoom

---

## 5. Preprocessing pipeline

1. Resize image to 300×300 pixels (EfficientNet-B3 native size)
2. Apply CLAHE (contrast enhancement) — improves retinal feature visibility
3. Normalize with ImageNet mean and std
4. Data augmentation during training only

---

## 6. Web application

**Endpoint:** POST /predict
**Input:** fundus image file (JPG or PNG)
**Output:** JSON with predicted grade, label, confidence scores
```json
{
  "grade": 2,
  "label": "Moderate DR",
  "confidence": 0.847,
  "all_scores": [0.02, 0.05, 0.85, 0.05, 0.03]
}
```

---

## 7. Evaluation metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct classifications |
| Quadratic weighted kappa | Official competition metric |
| Confusion matrix | Per-class breakdown |
| ROC curves | Per-class discrimination ability |

---

## 8. Development phases

| Phase | Description | Tools |
|---|---|---|
| 1 | Data exploration | Jupyter, Matplotlib |
| 2 | Model training | PyTorch, torchvision |
| 3 | Evaluation | scikit-learn, Matplotlib |
| 4 | Web app | Flask, HTML/CSS |
| 5 | Polish and deploy | GitHub, demo video |

---

## 9. Dataset download instructions

1. Go to kaggle.com/competitions/aptos2019-blindness-detection
2. Accept competition rules
3. Download train.zip and train.csv
4. Extract train/ folder into data/
5. Place train.csv into data/
6. Do not commit images to GitHub (see .gitignore)
```

Then create `.gitignore` in the root:
```
# Dataset and images
data/train_images/
*.png
*.jpg
*.jpeg

# Model weights
*.pth
*.pt

# Python cache
__pycache__/
*.pyc

# Jupyter checkpoints
.ipynb_checkpoints/
```