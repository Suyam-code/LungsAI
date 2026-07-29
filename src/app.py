import gradio as gr
import tensorflow as tf
import numpy as np
import os

# Resolve paths relative to this file rather than a fixed home directory, so
# the app runs from any working directory and on any machine after cloning.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_clean_model.keras")

CLASSES = ["COVID", "Lung_Cancer", "Normal", "Pneumonia"]

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded.")

def diagnostic_engine(input_img):
    if input_img is None:
        return "⚠️ No Image", "Please upload a chest X-ray.", {}

    # Preprocess
    img = input_img.resize((224, 224))
    img_array = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    conf = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

    covid_p = conf["COVID"]
    cancer_p = conf["Lung_Cancer"]
    normal_p = conf["Normal"]
    pneumonia_p = conf["Pneumonia"]

    top_class = CLASSES[np.argmax(preds)]
    final = top_class
    note = ""

    # Main decision logic
    if top_class == "Lung_Cancer":
        final = "Lung_Cancer"
        note = "Highest raw probability is Lung_Cancer."

    elif top_class == "Normal":
        # suspicious normal rule
        if cancer_p >= 0.15:
            final = "⚠️ Suspicious Normal / Possible Lung_Cancer"
            note = "Model predicts Normal, but Lung_Cancer probability is notable."
        else:
            final = "Normal"
            note = "Normal pattern is dominant and Lung_Cancer probability is low."

    elif top_class == "COVID":
        if cancer_p >= 0.20:
            final = "COVID / ⚠️ Possible Lung_Cancer"
            note = "COVID is top prediction, but Lung_Cancer probability is also meaningful."
        else:
            final = "COVID"
            note = "COVID pattern is dominant."

    elif top_class == "Pneumonia":
        if cancer_p >= 0.20:
            final = "Pneumonia / ⚠️ Possible Lung_Cancer"
            note = "Pneumonia is top prediction, but Lung_Cancer probability is also meaningful."
        else:
            final = "Pneumonia"
            note = "Pneumonia pattern is dominant."

    report = (
        f"Prediction: {final}\n"
        f"Top raw class: {top_class}\n"
        f"Top class confidence: {conf[top_class] * 100:.2f}%\n"
        f"Lung_Cancer probability: {cancer_p * 100:.2f}%\n"
        f"Normal probability: {normal_p * 100:.2f}%\n"
        f"COVID probability: {covid_p * 100:.2f}%\n"
        f"Pneumonia probability: {pneumonia_p * 100:.2f}%\n"
        f"Note: {note}"
    )

    return f"🩺 Diagnosis: {final}", report, conf

with gr.Blocks(title="LungsAI") as demo:
    gr.Markdown("# 🏥 LungsAI: Clinical Decision Support")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Chest X-ray")
            run_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            status = gr.Textbox(label="Diagnosis")
            details = gr.Textbox(label="Detailed Report", lines=8)
            chart = gr.Label(label="Probabilities", num_top_classes=4)

    run_btn.click(
        fn=diagnostic_engine,
        inputs=input_img,
        outputs=[status, details, chart]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)