import io
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image


LABELS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
IMAGE_SIZE = (224, 224)


# ✅ Patient Model
class Patient(BaseModel):
    name: str
    age: int
    gender: str
    blood: str
    diabetes_years: int
    symptoms: str


# ✅ Response Model
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


def _load_model() -> Optional[object]:
    global _model
    if _model is not None:
        return _model

    model_path = "model.keras"

    print("CURRENT DIR:", os.getcwd())
    print("FILES:", os.listdir())

    if not os.path.exists(model_path):
        print("❌ MODEL NOT FOUND")
        return None

    import tensorflow as tf
    _model = tf.keras.models.load_model(model_path, compile=False)

    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.") from e

    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get("/health")
def health():
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
        raise HTTPException(
            status_code=503,
            detail="Model not found. Place 'model.h5' in backend folder.",
        )

    image_bytes = await file.read()
    x = preprocess_image(image_bytes)

    try:
        preds = model.predict(x)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model prediction failed.") from e

    preds = np.array(preds).reshape(-1)

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = LABELS[idx]

    # 🔥 EXPLANATION WITH BULLET POINTS ONLY FOR RECOMMENDATION

    if label == "No_DR":
        explanation = f"""
No signs of diabetic retinopathy were detected.
Your retina appears healthy. This means there is currently low risk of vision damage.
However, since diabetes can affect eyes over time, regular checkups are important.
Recommendation:
• Maintain good blood sugar control
• Get eye checkup every 6–12 months
"""

    elif label == "Mild":
        explanation = f"""
Mild diabetic retinopathy detected.
This is an early stage where small changes are beginning in blood vessels.
Vision is usually not affected yet.
Recommendation:
• Monitor regularly
• Control blood sugar strictly
• Visit doctor if symptoms increase
"""

    elif label == "Moderate":
        explanation = f"""
Moderate diabetic retinopathy detected.
Blood vessels are becoming more damaged and may start affecting vision.
Recommendation:
• Consult an ophthalmologist soon
• Follow proper diabetes management
• Regular monitoring is required
"""

    elif label == "Severe":
        explanation = f"""
Severe diabetic retinopathy detected.
Significant damage is present and vision loss risk is high.
Recommendation:
• Immediate medical attention required
• Specialist consultation is necessary
"""

    else:
        explanation = f"""
Proliferative diabetic retinopathy detected.
This is an advanced stage where abnormal blood vessels grow in the retina.
Recommendation:
• Urgent treatment required
• Risk of blindness if untreated
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