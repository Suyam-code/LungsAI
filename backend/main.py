from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI(title="LungsAI API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.expanduser("~/Lung-Disease-Diagnostic-AI")
G1_PATH = os.path.join(BASE_DIR, "models/gatekeeper_1_simple.keras")
MAIN_PATH = os.path.join(BASE_DIR, "models/main_diagnostic_model_balanced.h5")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ---------- IMPORTANT ----------
# The order of this list must match the order used during training.
# If you see misclassifications, upload a known COVID X‑ray and look at the
# terminal output. The probabilities dictionary will show the correct order.
# Then adjust the list below accordingly.
CLASSES = ["COVID", "Lung_Cancer", "Normal", "Pneumonia"]
# -----------------------------

print("Loading models...")
g1 = tf.keras.models.load_model(G1_PATH, compile=False)
main_model = tf.keras.models.load_model(MAIN_PATH, compile=False)
print("Models loaded.")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.get("/")
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Gatekeeper 1 (X‑ray vs non‑X‑ray)
    g1_score = float(g1.predict(img_array, verbose=0).flatten()[0])
    print(f"[GK1] Score: {g1_score:.4f}")

    # Simple gatekeeper: low score = X‑ray, high = non‑X‑ray.
    # If your gatekeeper gives the opposite, change '>' to '<'.
    if g1_score > 0.5:
        return JSONResponse({
            "status": "error",
            "message": "Not a medical X‑ray. Please upload a chest X‑ray."
        })

    # Main prediction
    preds = main_model.predict(img_array, verbose=0)[0]
    probabilities = {cls: float(preds[i]) for i, cls in enumerate(CLASSES)}
    print(f"Probabilities: {probabilities}")

    # Simple argmax – highest probability wins
    final = CLASSES[np.argmax(preds)]

    return {
        "status": "success",
        "diagnosis": final,
        "confidence": probabilities[final] * 100,
        "probabilities": probabilities,
        "gatekeeper_score": float(g1_score)
    }