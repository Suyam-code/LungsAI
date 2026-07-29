# LungsAI — Multi-Class Chest X-Ray Diagnosis System

A deep-learning clinical decision-support tool that classifies a chest X-ray into
one of four classes — **COVID-19, Lung Cancer, Pneumonia, or Normal** — and
returns the prediction with a full probability breakdown.

Built with transfer learning on MobileNetV2 and trained on a balanced
20,000-image dataset.

**Live demo:** https://lungsai-bevbaa85pvrl5v9zf3anuh.streamlit.app

> **Not a medical device.** This is a student project built for learning. It must
> not be used for real diagnosis or clinical decisions.

*(The demo sleeps after 12 hours of inactivity on the free tier; the first load
after a nap takes a moment to wake up.)*

---

## ⚠️ Current status: v1 does not generalise, and I know why

I tested the trained model on a chest X-ray from outside my datasets and it
predicted the wrong class. Investigating it turned up two real methodology
problems, described in full under [Known
limitations](#known-limitations-what-i-found-and-why-it-matters). In short:

1. **The validation split leaks patients.** Images were split randomly by file,
   but the NIH data has multiple X-rays per patient (5,001 images from 2,521
   patients — one patient contributes 30). The same patient appears in both
   training and validation, so the reported accuracy is inflated.
2. **Each class comes from a different source dataset,** so the model can
   score well by recognising *which dataset* an image came from rather than the
   pathology in it.

**The 89.5% below is therefore not a trustworthy estimate of real performance.**
I am rebuilding this properly — see [Rebuild in progress](#rebuild-in-progress).

---

## Results (v1 — not trustworthy, see above)

| Metric | Score |
| --- | --- |
| Validation accuracy | 89.5% |
| Precision | 0.905 |
| Recall | 0.904 |
| F1-score | 0.904 |

Measured on a random 20% validation split of 20,001 images balanced across four
classes. Because that split leaks patients between train and validation, these
numbers overstate real-world performance and should be read as a baseline to
beat, not a result.

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
this problem, so `app.py` adds a decision layer: if the top prediction is
*Normal* but the Lung_Cancer probability is at or above 0.15, the result is
escalated to **"Suspicious Normal / Possible Lung_Cancer"** rather than being
reported as clear. The same idea applies when COVID or Pneumonia is the top class
and cancer probability is at or above 0.20 — the report flags both. The intent is
to surface uncertainty instead of hiding it behind a single label.

---

## Project structure

```
├── streamlit_app.py            # Streamlit app (this is what is deployed)
├── app.py                      # Gradio app: same logic, local entry point
├── src/
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

streamlit run streamlit_app.py     # Streamlit UI (this is what is deployed)
# or
python app.py                      # Gradio UI, same model and logic
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

## Known limitations: what I found and why it matters

I tested v1 on a chest X-ray from outside my training data and it returned the
wrong class. Rather than tune around it, I traced the cause. Three findings:

**1. Patient leakage between train and validation.**
`ImageDataGenerator(validation_split=0.2)` splits randomly *by file*. The NIH
images are named `patientID_scanNumber.png`, and my Lung_Cancer class holds 5,001
images from just 2,521 patients — one patient contributes 30 scans. So the same
patient's X-rays land in both training and validation, and the model can score
well by recognising anatomy it has already memorised. Any split for medical
imaging has to be grouped **by patient**, not by image.

**2. Each class comes from a different source dataset.**
COVID, Normal and Pneumonia come from the COVID-19 Radiography Database
(`COVID-1.png`, `Normal-1.png`, `Viral Pneumonia-1.png`); Lung_Cancer comes from
NIH ChestX-ray14 (`00000004_000.png`). Those datasets differ in scanner,
resolution, contrast normalisation and preprocessing — so a model can reach high
accuracy by learning *which dataset an image came from* rather than what is wrong
with the patient. This is shortcut learning, and it explains precisely why the
model works on held-out images from these datasets and fails on a real external
X-ray.

**3. The "Lung_Cancer" label is not a cancer diagnosis.**
NIH ChestX-ray14 has no lung-cancer label. It has **Mass** and **Nodule**, which
are radiological findings — many are benign. Labelling that class "Lung_Cancer"
overstates what the data supports.

Smaller issues: the base model is frozen (no fine-tuning), the 0.15 / 0.20
cancer-probability thresholds were chosen by inspection rather than from a
precision-recall curve, and there is no untouched test set — only train/val.

---

## Rebuild in progress

Fixing this properly means changing the problem, not just the hyperparameters:

- **Single source, three classes.** Rebuild using NIH ChestX-ray14 only —
  *No Finding / Pneumonia / Mass-Nodule* — so every class is drawn from the same
  distribution and the dataset shortcut disappears. This drops COVID, which only
  exists in a different dataset.
- **Patient-grouped splits** into train / validation / **test**, so no patient
  appears in more than one split.
- **Honest evaluation:** confusion matrix and per-class precision/recall, not a
  single aggregate number. I expect accuracy to fall — that drop is the point.
- **Grad-CAM** to confirm the model attends to lung fields rather than image
  borders or text markers. If the heatmaps sit on the edges, it is still cheating.
- **Threshold tuning** from a precision-recall curve on the held-out test set.

---

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · Gradio · FastAPI · NumPy · Pandas · Pillow
