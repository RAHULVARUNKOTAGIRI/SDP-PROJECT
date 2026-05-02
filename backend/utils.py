import os
from openai import OpenAI


def generate_explanation(
    label: str,
    confidence: float,
    age: int,
    gender: str,
    diabetes_years: int,
    symptoms: str
) -> str:
    """
    Generates a patient-friendly explanation using OpenAI.
    Includes patient details for better context.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    # 🔴 Fallback if API key not set
    if not api_key:
        return (
            f"Prediction: {label} ({confidence * 100:.1f}%). "
            f"Patient age {age}, diabetes duration {diabetes_years} years. "
            "This is an AI-based screening result, not a confirmed diagnosis. "
            "Please consult an eye specialist for proper medical evaluation."
        )

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are a medical assistant explaining diabetic retinopathy screening results.

Model Prediction:
- Label: {label}
- Confidence: {confidence * 100:.2f}%

Patient Details:
- Age: {age}
- Gender: {gender}
- Diabetes Duration: {diabetes_years} years
- Symptoms: {symptoms}

Explain clearly:
1. What this condition means (simple language)
2. Risk level (low / moderate / high) based on patient details
3. When the patient should consult a doctor

Important:
- Do NOT suggest medicines
- Do NOT say it is a confirmed diagnosis
- Keep explanation simple and short (5–8 lines)
"""

    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt,
        )

        return (response.output_text or "").strip()

    except Exception:
        return (
            "AI explanation could not be generated. "
            "Please consult an eye specialist for professional advice."
        )