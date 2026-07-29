# LungsAI — chest X-ray classification, and what happened when I checked my own numbers

This project started as a four-class chest X-ray classifier that reported **89.5%
validation accuracy**. Then I tested it on an X-ray from outside my datasets and
it got the answer wrong.

Investigating why turned the project into something more useful: a controlled
study of how patient leakage, dataset confounding, model architecture and
training-set size actually affect performance on chest radiographs. The honest
final model scores **63.65% accuracy / 0.689 ROC-AUC** on a binary task — far
below the original claim, and the only number here I trust.

> **Not a medical device.** A student project, not validated for clinical use.
> It must not be used to diagnose or treat anyone.

---

## The headline

| Version | Task | Test accuracy | Notes |
| --- | --- | --- | --- |
| v1 | 4-class | **89.5%** | Not trustworthy — leaked patients, confounded datasets |
| v2 | 3-class | 42.4% | Honest splits, frozen MobileNetV2 |
| v3 | 3-class | 41.0% | Fine-tuned MobileNetV2 — no better |
| v4 | 3-class | 44.3% | DenseNet121 (CheXNet architecture) — no better |
| v2 re-run | 3-class | 40.5% | Same code, different stack → **±2 point noise floor** |
| v5 | binary | **63.7%** (AUC 0.689) | Final model, all available data |

**The gap between 89.5% and reality is the point of this repository.**

---

## What was wrong with v1

**1. The validation split leaked patients.**
`ImageDataGenerator(validation_split=0.2)` splits randomly *by file*. NIH images
are named `patientID_scanNumber.png`, and my cancer class held 5,001 images from
just 2,521 patients — one patient contributed 30 scans. The same patient's X-rays
therefore appeared in both training and validation, letting the model score well
on anatomy it had already memorised. Medical imaging splits must be grouped **by
patient**.

**2. Each class came from a different source dataset.**
COVID, Normal and Pneumonia came from the COVID-19 Radiography Database
(`COVID-1.png`, `Normal-1.png`); the cancer class came from NIH ChestX-ray14
(`00000004_000.png`). Those datasets differ in scanner, resolution and contrast
normalisation, so a model can score highly by recognising *which dataset an image
came from* rather than the pathology in it. This is shortcut learning, and it
explains exactly why v1 worked on held-out images from these datasets and failed
on a real external X-ray.

**3. The "Lung_Cancer" label was not lung cancer.**
NIH ChestX-ray14 has no cancer label. It has **Mass** and **Nodule** —
radiological findings, many of them benign. The class name overstated what the
data supported. v2 onward calls it `Mass_Nodule`.

---

## The rebuild

Every subsequent version uses **NIH ChestX-ray14 only**, with splits grouped by
patient. `src/build_dataset_v2.py` and `src/build_dataset_2class.py` both assert
zero patient overlap between train, validation and test, and refuse to write the
dataset if that check fails. The test set uses NIH's official split list, which is
patient-disjoint by construction.

### Experiment 1 — does architecture matter? (3-class)

Three configurations on identical patient-disjoint splits:

| Model | Test accuracy |
| --- | --- |
| MobileNetV2, frozen | 42.41% |
| MobileNetV2, fine-tuned | 40.96% |
| DenseNet121, fine-tuned | 44.32% |

Chance is 33.3%, and the standard error at n=1311 is ~1.37 points, so none of
these differ meaningfully. Fine-tuning *hurt* slightly — training accuracy rose to
0.51 while validation stayed at 0.44, meaning the extra capacity bought fit
without generalisation.

### Measuring the noise floor

Re-running the v2 baseline on a different TensorFlow version gave **40.50%**
instead of 42.41% — same code, same data, same seed. **A 1.9-point swing between
two runs of identical code.**

That is as large as the spread between the three architectures above, which
retroactively settles the question: the architecture differences were never real.
Quoting accuracy without knowing its run-to-run variance is how people convince
themselves of things the data does not support.

### Experiment 2 — does data volume matter? (binary)

Switching to a binary task (No Finding vs Mass/Nodule) unlocked far more data.
Training size was varied while **the validation and test sets were held
byte-identical**, so data volume was the only variable:

| Train images/class | Accuracy | ROC-AUC |
| --- | --- | --- |
| 689 | 57.25% | 0.596 |
| 5,000 | 63.40% | 0.686 |
| 6,880 (all available) | 63.65% | 0.689 |

**Data was the bottleneck — up to a point.** Going 689 → 5,000 (7.3×) gained
+0.09 AUC, far beyond the noise floor. Going 5,000 → 6,880 (1.4×) gained nothing.
The curve saturates around 5,000 images per class for this configuration.

---

## Final model

Binary **No Finding vs Mass/Nodule**, MobileNetV2 with a frozen backbone, trained
on all 6,880 available images per class.

| Metric | Value |
| --- | --- |
| Test accuracy | 63.65% (± 1.08 SE, n=2000) |
| ROC-AUC | 0.689 |
| Chance | 50.0% |

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Mass_Nodule | 0.612 | 0.748 | 0.673 |
| No_Finding | 0.676 | 0.525 | 0.591 |

Evaluated on 2,000 images from 867 patients that appear in no other split. For
context, CheXNet-class models trained on the full NIH dataset report AUC around
0.75–0.78 for Mass and Nodule, so this sits below the published range — as
expected given a frozen backbone and a fraction of the data.

---

## What I'd do next

- **Fine-tune at the larger data scale.** Fine-tuning failed at v3, but that was
  on 689 images per class, where the extra capacity had nothing to learn from.
  With 6,880 per class it may now pay off. Capacity and data interact, and I only
  varied one at a time.
- **Grad-CAM overlays** to confirm the model attends to lung fields rather than
  image borders or text markers. Without this I cannot rule out a subtler
  shortcut.
- **Multi-label training** on all 14 NIH findings, which is how CheXNet is
  actually trained and uses far more of the data than a balanced binary subset.
- **Tune the decision threshold** from a precision-recall curve rather than
  defaulting to argmax — recall on No_Finding (0.525) is notably weaker than on
  Mass_Nodule (0.748), so the operating point is not balanced.

---

## Project structure

```
├── streamlit_app.py                 # Deployed demo
├── app.py                           # Gradio version, same logic
├── src/
│   ├── build_dataset_v2.py          # NIH 3-class, patient-grouped splits
│   ├── build_dataset_2class.py      # NIH binary, variable training size
│   ├── train_v2.py                  # Frozen MobileNetV2 baseline
│   ├── train_v3_finetune.py         # Two-stage fine-tuning
│   ├── train_v4_densenet.py         # DenseNet121 control
│   └── train_2class.py              # Binary, data-volume experiment
├── models/                          # Trained models + metrics JSON per run
└── data/                            # Not committed (see below)
```

Every training script writes a metrics JSON with the confusion matrix, per-class
scores, standard error and chance baseline, so each result is reproducible and
auditable.

---

## Running it

Requires Python 3.11 and the NIH ChestX-ray14 dataset (images plus
`Data_Entry_2017.csv` and `test_list_NIH.txt`) under `data/raw/nih_resized/`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/build_dataset_2class.py --per-class 6880
python src/train_2class.py --per-class 6880

streamlit run streamlit_app.py
```

On Apple Silicon, `tensorflow-metal` must match your TensorFlow version — the
plugin lags TF releases, and a mismatch produces a `dlopen` failure at import.
TF 2.16.2 with tensorflow-metal 1.2.0 works.

---

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · DenseNet121 · scikit-learn ·
Streamlit · Gradio · NumPy · Pandas
