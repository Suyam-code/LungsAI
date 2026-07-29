# LungsAI — Multi-Class Chest X-Ray Diagnosis System

A deep-learning clinical decision-support tool that classifies a chest X-ray into
one of four classes — **COVID-19, Lung Cancer, Pneumonia, or Normal** — and
returns the prediction with a full probability breakdown.

Built with transfer learning on MobileNetV2, trained on a balanced 20,000-image
dataset, and served through a Gradio interface.

> **Not a medical device.** This is a student project built for learning. It must
> not be used for real diagnosis or clinical decisions.

---

## Results

| Metric | Score |
| --- | --- |
| Validation accuracy | 89.5% |
| Precision | 0.905 |
| Recall | 0.904 |
| F1-score | 0.904 |

Trained on 20,001 images balanced across the four classes (~5,000 each).

---

## How it works

**Model.** MobileNetV2 pre-trained on ImageNet, used as a frozen feature
extractor, with a custom head: global average pooling → batch normalization →
dense(128, ReLU) → dropout(0.5) → dense(4, softmax). Trained with Adam at a
1e-4 learning rate, early stopping, and learning-rate reduction on plateau.

**Handling the risk asymmetry.** Missing a lung cancer case is far more costly
than a false alarm, so the training applies a **class weight of 1.8 to
Lung_Cancer** while the other three classes stay at 1.0. This deliberately
biases the model toward catching cancer at the cost of some false positives.

**Safety logic on top of the model.** Raw argmax alone is not safe enough for
this problem, so `src/app.py` adds a decision layer: if the top prediction is
*Normal* but the Lung_Cancer probability is at or above 0.15, the result is
escalated to **"Suspicious Normal / Possible Lung_Cancer"** rather than being
reported as clear. The same idea applies when COVID or Pneumonia is the top class
and cancer probability is at or above 0.20 — the report flags both. The intent is
to surface uncertainty instead of hiding it behind a single label.

---

## Project structure

```
├── src/
│   ├── app.py                  # Gradio app: upload an X-ray, get a report
│   ├── train_clean_model.py    # Training script (MobileNetV2 + class weights)
│   ├── check_data.py           # Dataset sanity checks
│   └── check_data_vis.py       # Visual inspection of samples
├── scripts/                    # Dataset preparation and augmentation
│   ├── copy_lung_data.py
│   ├── augment_covid.py
│   └── augment_pneumonia.py
├── backend/main.py             # Optional FastAPI service (alternative to Gradio)
├── frontend/                   # Minimal HTML/CSS/JS client for the FastAPI backend
├── models/
│   └── final_clean_model.keras # Trained model (committed, ~12 MB)
└── data/                       # Not committed — see "Getting the data" below
```

Additional root-level scripts (`add_chexpert*.py`, `balance_to_10000.py`,
`remove_lateral.py`, `test_cancer*.py`) are one-off data-preparation and
spot-check utilities kept for reference.

---

## Running it

Requires Python 3.10+.

```bash
git clone https://github.com/Suyam-code/LungsAI.git
cd LungsAI

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/app.py
```

Then open the local URL Gradio prints (default `http://127.0.0.1:7860`), upload a
chest X-ray, and click **Analyze**.

The trained model is included in the repo, so this works without downloading the
dataset or retraining.

---

## Getting the data (only needed to retrain)

The image data is not committed — it is several gigabytes and comes from public
sources:

- **COVID-19 Radiography Database** (COVID, Normal, Pneumonia) — available on Kaggle
- **NIH Chest X-ray Dataset** — used for additional Normal and abnormal samples

Arrange the images into this layout before training:

```
data/processed_clean/
├── COVID/
├── Lung_Cancer/
├── Normal/
└── Pneumonia/
```

The scripts in `scripts/` handle copying, balancing, and augmenting the classes
to roughly 5,000 images each. Then train with:

```bash
python src/train_clean_model.py
```

The best checkpoint by validation accuracy is written to
`models/final_clean_model.keras`.

---

## What I'd improve next

- **Fine-tune the base model.** MobileNetV2 is frozen; unfreezing the top layers
  at a low learning rate would likely add a few points of accuracy.
- **Report per-class metrics.** The headline numbers are aggregate — a confusion
  matrix and per-class recall would show exactly where the model confuses
  Pneumonia with COVID.
- **Validate the thresholds.** The 0.15 and 0.20 cancer-probability cut-offs were
  chosen by inspection, not tuned. They should be set from a
  precision-recall curve on a held-out set.
- **Add a proper test set.** Currently there is a train/validation split only; a
  third untouched split would give a more honest performance estimate.
- **Grad-CAM overlays** to show which region of the X-ray drove the prediction —
  important for any tool meant to support a clinician rather than replace one.

---

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · Gradio · FastAPI · NumPy · Pandas · Pillow
