# Retinal Disease Detector

A deep learning web application that detects diabetic retinopathy severity
from fundus retinal images using EfficientNet-B3, achieving medical-grade
classification across 5 severity levels.

>  Project status: In progress — Phase 1 (model training)

---

## What it does

A user uploads a fundus retinal image through the web interface. The system
preprocesses the image, passes it through a fine-tuned EfficientNet-B3 model,
and returns a diabetic retinopathy severity grade (0–4) with confidence scores
for each class — in under 2 seconds.

---

## Severity grades

| Grade | Label | Description |
|---|---|---|
| 0 | No DR | Healthy retina |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than mild, less than severe |
| 3 | Severe | Extensive damage, no proliferative signs |
| 4 | Proliferative DR | Most severe, neovascularization present |

---

## System overview

*Diagram coming soon*

---

## Tech stack

| Layer | Technology |
|---|---|
| Model architecture | EfficientNet-B3 (pretrained ImageNet) |
| Deep learning | PyTorch, torchvision |
| Image processing | OpenCV, PIL |
| Web backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Dataset | APTOS 2019 Blindness Detection (Kaggle) |

---

## Project structure
```
retinal-disease-detector/
├── model/
│   ├── train.py
│   ├── dataset.py
│   └── evaluate.py
├── app/
│   ├── app.py
│   ├── static/
│   └── templates/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── data/
│   └── (download instructions in docs/SPEC.md)
├── docs/
│   └── SPEC.md
└── README.md
```
## 🚀 Live Demo

**Try it here:** https://huggingface.co/spaces/hidayet-yaakoubi/retinal-disease-detector

Upload any fundus retinal image and get an instant diabetic retinopathy severity grade.

---

## Progress log

- [x] Project defined and documented
- [x] Data exploration
- [x] Model training — kappa 0.9053
- [x] Model evaluation
- [x] Web app — working locally
- [x] Deployed online — live public demo
- [ ] Demo video

---
## 📄 Full Project Report

For a detailed explanation of the dataset, model architecture, training methodology, and results, see the [full project report](docs/Retinal_Disease_Detector_Report.pdf).

---

## Author

**Hidayet Allah Yaakoubi**
Engineering student — Tunisia