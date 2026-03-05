"""
SHL Assessment Recommendation API + Frontend
FastAPI backend with /health, /recommend endpoints
Serves the SHL-styled HTML frontend at /

Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from recommender import get_recommender

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends SHL assessments based on job descriptions or natural language queries",
    version="1.0.0"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request / Response Models ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str

class AssessmentResult(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int]
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentResult]

# ── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the SHL-styled HTML frontend"""
    frontend_path = Path(__file__).parent / "frontend.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found. Place frontend.html in the same directory.</h1>", status_code=404)

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Recommend SHL assessments for a given job description or query.

    - Accepts natural language query, job description text, or URL
    - Returns 5-10 most relevant Individual Test Solutions
    - Balances technical and behavioral assessments when needed
    """
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    raw = request.query.strip()

    # If the input looks like a URL, fetch the page and extract text
    if raw.lower().startswith(("http://", "https://")):
        try:
            resp = requests.get(raw, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            # Very simple heuristic: join visible text paragraphs
            texts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            extracted = "\n".join(t for t in texts if len(t) > 40)
            if extracted:
                raw = extracted
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    try:
        rec = get_recommender()
        results = rec.recommend(raw, n=10)
        return RecommendResponse(recommended_assessments=results)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Recommender not initialized: {str(e)}. Run scraper.py and build_embeddings.py first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/api")
def root():
    return {
        "message": "SHL Assessment Recommendation API",
        "endpoints": {
            "frontend": "GET /",
            "health": "GET /health",
            "recommend": "POST /recommend"
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
