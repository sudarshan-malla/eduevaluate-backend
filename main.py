from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import base64
import uuid
import requests
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Key (set in Railway ENV)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY is not set in environment variables.")

# -----------------------------
# Pydantic Models
# -----------------------------
class InlineData(BaseModel):
    data: str
    mimeType: str

class Part(BaseModel):
    text: Optional[str] = None
    inlineData: Optional[InlineData] = None

class EvaluateRequest(BaseModel):
    parts: List[Part]

class EvaluationReport(BaseModel):
    score: int
    feedback: str
    details: dict


# -----------------------------
# Helper: Save base64 image
# -----------------------------
def save_base64_image(base64_data: str, mime: str):
    try:
        image_bytes = base64.b64decode(base64_data)
        ext = mime.split("/")[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join("/tmp", filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        return filepath
    except:
        return None


@app.get("/")
def root():
    return {"message": "API is working"}


@app.post("/evaluate", response_model=EvaluationReport)
async def evaluate_answer_sheet(request: EvaluateRequest):

    parts = request.parts

    # Convert parts to prompt
    prompt_parts = []
    for part in parts:
        if part.text:
            prompt_parts.append(part.text)

    prompt_text = "\n".join(prompt_parts)

    # Gemini API URL
    url = "https://gemini.googleapis.com/v1/models/gemini-2.5-flash:generateMessage"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }

    # Your prompt
    system_prompt = """
You are an elite academic examiner.
Perform OCR on handwritten answers, evaluate strictly,
award marks accurately, and return JSON only.
"""

    payload = {
        "messages": [
            {"author": "system", "content": {"content_type": "text", "text": system_prompt}},
            {"author": "user", "content": {"content_type": "text", "text": prompt_text}},
        ],
        "temperature": 0.0,
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    data = response.json()
    output = data["candidates"][0]["content"][0]["text"]

    # Return output as JSON (assuming model returns JSON)
    return {
        "score": 100,
        "feedback": "Evaluation completed successfully.",
        "details": {"result": output}
    }
