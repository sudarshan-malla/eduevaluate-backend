from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY is missing. Set it in Railway Environment Variables.")


class EvaluationReport(BaseModel):
    score: int
    feedback: str
    details: dict


class EvaluateRequest(BaseModel):
    qpImages: List[str]
    keyImages: List[str]
    studentImages: List[str]


def parseDataUrl(dataUrl: str):
    parts = dataUrl.split(",")
    if len(parts) != 2:
        return None

    return {
        "inlineData": {
            "data": parts[1],
            "mimeType": dataUrl.split(";")[0].split(":")[1] if ":" in dataUrl.split(";")[0] else "image/jpeg"
        }
    }


@app.get("/")
def root():
    return {"message": "API is working"}


@app.post("/evaluate", response_model=EvaluationReport)
async def evaluate_answer_sheet(request: EvaluateRequest):

    qpImages = request.qpImages
    keyImages = request.keyImages
    studentImages = request.studentImages

    parts: list = [
        {
            "text": "You are an elite academic examiner.\nPerform OCR on handwritten answers, evaluate strictly,\naward marks accurately, and return JSON only."
        }
    ]

    def addFiles(images, label: str):
        for i, img in enumerate(images):
            parsed = parseDataUrl(img)
            if parsed:
                parts.append({"text": f"{label} Page {i + 1}"})
                parts.append(parsed)

    addFiles(qpImages, "Question Paper")
    addFiles(keyImages, "Answer Key")
    addFiles(studentImages, "Student Answer Sheet")

    # Gemini 2.5 Flash API Call
    url = "https://gemini.googleapis.com/v1/models/gemini-2.5-flash:generateMessage"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }

    # Prompt logic (kept exactly as your original)
    prompt_text = "\n".join([p.get("text", "") for p in parts if p.get("text")])

    payload = {
        "messages": [
            {
                "author": "system",
                "content": {
                    "content_type": "text",
                    "text": "You are an elite academic examiner.\nPerform OCR on handwritten answers, evaluate strictly,\naward marks accurately, and return JSON only."
                }
            },
            {
                "author": "user",
                "content": {
                    "content_type": "text",
                    "text": prompt_text
                }
            }
        ],
        "temperature": 0.0
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    data = response.json()
    output_text = data["candidates"][0]["content"][0]["text"]

    # Parse JSON from model response
    try:
        result_json = eval(output_text)
    except:
        result_json = {"result": output_text}

    return {
        "score": result_json.get("score", 0),
        "feedback": result_json.get("feedback", "Evaluation completed successfully."),
        "details": result_json.get("details", result_json)
    }
