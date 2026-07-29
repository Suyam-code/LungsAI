"""
LungsAI demo - binary chest X-ray classification (No Finding vs Mass/Nodule).

This serves the v5 model: MobileNetV2 with a frozen backbone, trained on 6,880
NIH ChestX-ray14 images per class with patient-disjoint splits. Measured
performance on 2,000 held-out images from unseen patients:

    accuracy 63.65%  |  ROC-AUC 0.689  |  chance 50%

An earlier four-class version of this project reported 89.5% accuracy, but that
number came from a split which leaked patients between train and validation, and
from classes drawn from different source datasets. It is not served here. The
full analysis is in the README.
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "binary_6880.keras")
METRICS_PATH = os.path.join(BASE_DIR, "models", "binary_6880_metrics.json")

# Order must match training. flow_from_directory sorts alphabetically, which
# gave {'Mass_Nodule': 0, 'No_Finding': 1} - getting this wrong silently
# inverts every prediction.
CLASSES = ["Mass_Nodule", "No_Finding"]
LABELS = {"Mass_Nodule": "Mass or nodule visible", "No_Finding": "No finding"}

REPO = "https://github.com/Suyam-code/LungsAI"

st.set_page_config(page_title="LungsAI", page_icon="🫁", layout="centered")


@st.cache_resource
def load_model():
    """
    Load once and reuse. Streamlit re-runs the whole script on every
    interaction, so without this the model would be re-read from disk on every
    click - slow, and a quick way to exhaust memory on a 1 GB instance.
    """
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def classify(image: Image.Image):
    """
    Preprocess exactly as in training: 224x224 RGB through MobileNetV2's
    preprocess_input, which scales to [-1, 1]. Using a plain /255 rescale here
    instead would not crash - it would just quietly degrade every prediction,
    which is the kind of mismatch that is very hard to notice after the fact.
    """
    model = load_model()

    img = image.resize((224, 224)).convert("RGB")
    arr = preprocess_input(np.array(img, dtype=np.float32))
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    return {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}


# ------------------------------------------------------------------ UI

st.title("LungsAI — chest X-ray screening demo")
st.write(
    "Upload a chest X-ray. The model estimates whether a **mass or nodule** is "
    "visible, or whether the image shows **no finding**."
)

st.error(
    "**Not a medical device.** This is a student machine-learning project. It is "
    "not validated for clinical use and must not be used to diagnose, treat, or "
    "make any decision about a real patient. If you have a health concern, speak "
    "to a qualified doctor."
)

metrics = load_metrics()
if metrics:
    acc = metrics["test_accuracy"] * 100
    auc = metrics["roc_auc"]
    st.warning(
        f"**This model is only modestly better than guessing.** On 2,000 held-out "
        f"images from patients it never saw during training it scores "
        f"**{acc:.1f}% accuracy (ROC-AUC {auc:.3f})** against a 50% chance "
        f"baseline. Treat individual predictions as unreliable. "
        f"An earlier version of this project claimed 89.5% accuracy; that figure "
        f"came from a split that leaked patients between training and validation "
        f"and is not reproducible — the [README]({REPO}) documents how I found "
        f"that and what I did about it."
    )

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded)
    col_img, col_result = st.columns(2)

    with col_img:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col_result:
        with st.spinner("Analysing…"):
            conf = classify(image)

        top = max(conf, key=conf.get)
        st.info(f"**{LABELS[top]}**")

        st.write("**Model output**")
        for cls in CLASSES:
            st.progress(conf[cls], text=f"{LABELS[cls]}: {conf[cls] * 100:.1f}%")

        # These probabilities are raw softmax outputs and have not been
        # calibrated, so a "70%" here does not mean the model is right 70% of
        # the time. Saying so is more useful than implying false precision.
        st.caption(
            "Probabilities are uncalibrated softmax outputs — they indicate "
            "relative preference, not true confidence."
        )

st.divider()
st.caption(
    "v5: MobileNetV2 (frozen backbone), trained on 6,880 NIH ChestX-ray14 images "
    f"per class with patient-disjoint splits. [Source and full write-up]({REPO})"
)
