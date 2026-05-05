import io
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Suppress TensorFlow CPU instruction warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

LABELS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
IMAGE_SIZE = (224, 224)


class Patient(BaseModel):
    name: str
    age: int
    gender: str
    blood: str
    diabetes_years: int
    symptoms: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    patient: Patient
    explanation: str


app = FastAPI(title="Diabetic Retinopathy Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model = None


def _build_model():
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def _load_model() -> Optional[object]:
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf

    # Try loading full saved model first
    for model_path in ["model.keras", "model.h5"]:
        if os.path.exists(model_path):
            try:
                _model = tf.keras.models.load_model(model_path)
                print(f"✅ MODEL LOADED FROM {model_path}")
                return _model
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")

    # Fallback: try weights file
    weights_path = "model_weights.weights.h5"
    if os.path.exists(weights_path):
        try:
            _model = _build_model()
            _model.load_weights(weights_path)
            print("✅ MODEL LOADED FROM WEIGHTS")
            return _model
        except Exception as e:
            print("❌ MODEL LOAD ERROR:", e)

    print("❌ NO MODEL FILE FOUND")
    return None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    print("IMAGE SHAPE:", arr.shape)
    return arr


# Root route — fixes 404 on GET /
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Diabetic Retinopathy Detection API is running",
        "docs": "/docs",
        "health": "/health"
    }


# Health route — supports both GET and HEAD (fixes 405 error)
@app.api_route("/health", methods=["GET", "HEAD"])
def health(request: Request):
    model = _load_model()
    return {"status": "ok", "model_loaded": bool(model)}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    diabetes_years: int = Form(...),
    blood: str = Form(...),
    symptoms: str = Form("")
):
    if not file:
        raise HTTPException(status_code=400, detail="No image uploaded.")

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG/PNG/WebP image.")

    model = _load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    image_bytes = await file.read()
    x = preprocess_image(image_bytes)

    try:
        preds = model.predict(x)
        preds = np.array(preds).reshape(-1)
        print("PREDICTIONS:", preds)
    except Exception as e:
        print("❌ PREDICTION ERROR:", e)
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = LABELS[idx]

    if label == "No_DR":
        explanation = """
No signs of diabetic retinopathy were detected.

Your retina appears healthy. This means there is currently low risk of vision damage.
However, since diabetes can affect eyes over time, regular checkups are important.

Recommendation:
- Maintain good blood sugar control
- Get eye checkup every 6–12 months
"""
    elif label == "Mild":
        explanation = """
Mild diabetic retinopathy detected.

This is an early stage where small changes are beginning in blood vessels.
Vision is usually not affected yet.

Recommendation:
- Monitor regularly
- Control blood sugar strictly
- Visit doctor if symptoms increase
"""
    elif label == "Moderate":
        explanation = """
Moderate diabetic retinopathy detected.

Blood vessels are becoming more damaged and may start affecting vision.

Recommendation:
- Consult an ophthalmologist soon
- Follow proper diabetes management
- Regular monitoring is required
"""
    elif label == "Severe":
        explanation = """
Severe diabetic retinopathy detected.

Significant damage is present and vision loss risk is high.

Recommendation:
- Immediate medical attention required
- Specialist consultation is necessary
"""
    else:
        explanation = """
Proliferative diabetic retinopathy detected.

This is an advanced stage where abnormal blood vessels grow in the retina.

Recommendation:
- Urgent treatment required
- Risk of blindness if untreated
"""

    return PredictResponse(
        label=label,
        confidence=confidence,
        patient={
            "name": name,
            "age": age,
            "gender": gender,
            "blood": blood,
            "diabetes_years": diabetes_years,
            "symptoms": symptoms
        },
        explanation=explanation
    )