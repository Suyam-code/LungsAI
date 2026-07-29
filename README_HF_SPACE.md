---
title: LungsAI
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# LungsAI — Chest X-Ray Classification Demo

Classifies a chest X-ray into **COVID-19, Lung Cancer, Pneumonia, or Normal**
using MobileNetV2 transfer learning, and returns the full probability breakdown.

**Not a medical device.** This is a student project built for learning. It is not
validated for clinical use and must not be used to diagnose or treat anyone.

## Details

- **Validation accuracy:** 89.5% (precision 0.905, recall 0.904, F1 0.904)
- **Training data:** 20,001 images balanced across four classes (~5,000 each),
  from the COVID-19 Radiography Database and the NIH Chest X-ray Dataset
- **Model:** MobileNetV2 (ImageNet weights, frozen) + global average pooling →
  batch norm → dense(128) → dropout(0.5) → dense(4, softmax)
- **Safety logic:** Lung_Cancer is trained with a 1.8 class weight, and the app
  escalates a *Normal* prediction to "Suspicious Normal" when cancer probability
  is at or above 0.15 — the goal is to surface uncertainty rather than hide it

Full write-up, training code, and known limitations:
**[github.com/Suyam-code/LungsAI](https://github.com/Suyam-code/LungsAI)**
