from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import httpx
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for now, restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")


# ----------------------------
# Request model (EXACT MATCH)
# ----------------------------

class GeminiRequest(BaseModel):
    parts: List[Dict[str, Any]]


# ----------------------------
# Endpoint
# ----------------------------

@app.post("/evaluate")
async def evaluate_answer_sheet(payload: GeminiRequest):
    """
    NOTHING is generated here.
    Prompt + images already come from frontend inside `parts`.
    This function only forwards them to Gemini.
    """

    try:
        gemini_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": payload.parts  # 🔴 PROMPT IS HERE
                }
            ]
        }

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent",
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json=gemini_payload
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
