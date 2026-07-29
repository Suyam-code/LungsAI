"""
Streamlit version of the LungsAI demo, for deployment on Streamlit Community Cloud.

The model loading, preprocessing, and decision logic are identical to app.py
(the Gradio version) - only the interface layer differs. Keeping the logic the
same means both entry points behave identically.
"""

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_clean_model.keras")

CLASSES = ["COVID", "Lung_Cancer", "Normal", "Pneumonia"]

st.set_page_config(page_title="LungsAI", page_icon="🫁", layout="centered")


@st.cache_resource
def load_model():
    """
    Load the model once and reuse it across reruns.

    Streamlit re-executes the whole script on every interaction, so without
    @st.cache_resource the model would be re-read from disk on every click -
    slow, and a fast way to exhaust memory on a 1 GB instance.
    """
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def classify(image: Image.Image):
    """Preprocess to the 224x224 RGB, [0,1]-scaled input used in training."""
    model = load_model()

    img = image.resize((224, 224))
    img_array = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    conf = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    top_class = CLASSES[int(np.argmax(preds))]
    return top_class, conf


def apply_decision_logic(top_class, conf):
    """
    Decision layer on top of raw argmax.

    Missing a lung cancer case is far more costly than a false alarm, so a
    notable cancer probability is surfaced even when another class wins the
    argmax. Thresholds were chosen by inspection and are not yet tuned - see
    the README's limitations section.
    """
    cancer_p = conf["Lung_Cancer"]

    if top_class == "Lung_Cancer":
        return "Lung_Cancer", "Highest raw probability is Lung_Cancer.", "error"

    if top_class == "Normal":
        if cancer_p >= 0.15:
            return (
                "Suspicious Normal / Possible Lung_Cancer",
                "Model predicts Normal, but Lung_Cancer probability is notable.",
                "warning",
            )
        return "Normal", "Normal pattern is dominant and Lung_Cancer probability is low.", "success"

    if top_class == "COVID":
        if cancer_p >= 0.20:
            return (
                "COVID / Possible Lung_Cancer",
                "COVID is top prediction, but Lung_Cancer probability is also meaningful.",
                "warning",
            )
        return "COVID", "COVID pattern is dominant.", "error"

    # Pneumonia
    if cancer_p >= 0.20:
        return (
            "Pneumonia / Possible Lung_Cancer",
            "Pneumonia is top prediction, but Lung_Cancer probability is also meaningful.",
            "warning",
        )
    return "Pneumonia", "Pneumonia pattern is dominant.", "error"


# ---------------- UI ----------------

st.title("LungsAI — Chest X-Ray Classification Demo")
st.write(
    "Upload a chest X-ray and the model classifies it as COVID-19, Lung Cancer, "
    "Pneumonia, or Normal, with the full probability breakdown."
)

# Shown before any interaction, so nobody uses this without seeing it.
st.error(
    "**Not a medical device.** This is a student machine-learning project built for "
    "learning purposes. It is not validated for clinical use and must not be used to "
    "diagnose, treat, or make any decision about a real patient. If you have a health "
    "concern, please speak to a qualified doctor."
)

# Honesty about the model's actual reliability. The v1 training split leaked
# patients and each class came from a different source dataset, so the model
# often fails on X-rays from outside those datasets. Documented rather than
# hidden - see the README for the full analysis and the rebuild plan.
st.warning(
    "**Known issue — v1 does not generalise.** Testing revealed two methodology "
    "problems: the train/validation split leaked patients (multiple scans of the "
    "same person on both sides), and each class was drawn from a different source "
    "dataset, so the model partly learned to recognise the dataset rather than the "
    "pathology. It frequently misclassifies X-rays from outside its training "
    "sources. A rebuild on a single dataset with patient-grouped splits is in "
    "progress — the analysis is written up in the "
    "[README](https://github.com/Suyam-code/LungsAI#known-limitations-what-i-found-and-why-it-matters)."
)

uploaded = st.file_uploader(
    "Upload a chest X-ray", type=["png", "jpg", "jpeg"], label_visibility="visible"
)

if uploaded is not None:
    image = Image.open(uploaded)

    col_img, col_result = st.columns(2)

    with col_img:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col_result:
        with st.spinner("Analyzing…"):
            top_class, conf = classify(image)
            final, note, severity = apply_decision_logic(top_class, conf)

        if severity == "success":
            st.success(f"**Result: {final}**")
        elif severity == "warning":
            st.warning(f"**Result: {final}**")
        else:
            st.error(f"**Result: {final}**")

        st.caption(note)

        st.write("**Probabilities**")
        for cls in CLASSES:
            st.progress(conf[cls], text=f"{cls}: {conf[cls] * 100:.2f}%")

        st.caption(f"Top raw class: {top_class}")

st.divider()
st.caption(
    "Model: MobileNetV2 (transfer learning) on a balanced 20,000-image dataset. "
    "The 89.5% figure reported during training came from a split that leaked "
    "patients, so it overstates real performance — see the README for the full "
    "write-up. "
    "[Source code](https://github.com/Suyam-code/LungsAI)"
)
