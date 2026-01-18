from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import requests

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Key (Railway env var)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY is missing. Set it in Railway Environment Variables.")

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


@app.get("/")
def root():
    return {"message": "API is working"}


@app.post("/evaluate", response_model=EvaluationReport)
async def evaluate_answer_sheet(request: EvaluateRequest):

    parts = request.parts

    # Build the prompt from parts
    prompt_parts = []
    for part in parts:
        if part.text:
            prompt_parts.append(part.text)

    prompt_text = "\n".join(prompt_parts)

    # Gemini API endpoint
    url = "https://gemini.googleapis.com/v1/models/gemini-2.5-flash:generateMessage"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }

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

    # Extract output text
    output_text = data["candidates"][0]["content"][0]["text"]

    # NOTE:
    # This assumes Gemini returns JSON in output_text.
    # If it returns plain text, you must parse it.

    return {
        "score": 100,
        "feedback": "Evaluation completed successfully.",
        "details": {"result": output_text}
    }
