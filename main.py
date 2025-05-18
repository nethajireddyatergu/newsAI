import os
import requests
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_SUMMARIZER = "facebook/bart-large-cnn"
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# --- Initialize app ---
app = FastAPI(title="Web Summarizer & News API")

# --- Pydantic models ---
class URLRequest(BaseModel):
    url: str
    summary_length: str = "short"  # "short" or "detailed"

# --- Utilities ---
def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def scrape_website(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        if not text.strip():
            raise ValueError("No readable content found.")
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {e}")

def summarize_with_huggingface(text: str, max_length: int, min_length: int) -> str:
    url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_SUMMARIZER}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": False
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Hugging Face error: {response.json()}")
    result = response.json()
    return result[0]["summary_text"]

# --- Routes ---

@app.post("/summarize-url/")
async def summarize_url(request: URLRequest):
    full_text = scrape_website(request.url)
    chunks = chunk_text(full_text)

    summaries = []
    for chunk in chunks:
        summary = summarize_with_huggingface(
            chunk,
            max_length=80 if request.summary_length == "short" else 150,
            min_length=30
        )
        summaries.append(summary)

    combined = " ".join(summaries)
    final_summary = summarize_with_huggingface(
        combined,
        max_length=150 if request.summary_length == "short" else 250,
        min_length=50
    )

    return {"summary": final_summary}

@app.get("/news/{sector}")
async def get_news_by_sector(sector: str):
    url = f"https://gnews.io/api/v4/search?q={sector}&lang=en&country=in&apikey={GNEWS_API_KEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    if "articles" in data and data["articles"]:
        return {"sector": sector, "articles": data["articles"]}
    return {"sector": sector, "message": "No articles found."}
