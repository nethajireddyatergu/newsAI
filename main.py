# main.py
import httpx
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


app = FastAPI()
load_dotenv()  # This is for local dev; won't affect production

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
API_KEY = os.getenv("GNEWS_API_KEY")

model = genai.GenerativeModel('gemini-1.5-pro') 
app = FastAPI()

# Request model
class URLRequest(BaseModel):
    url: str
    summary_length: str = "short summary"  # default value

# Text chunking
def chunk_text(text, chunk_size=3000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Scrape and summarize
def scrape_and_summarize(url: str, summary_length: str = "short summary"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)

        if not text.strip():
            raise ValueError("No readable content found on the page.")

        # Chunk and summarize
        chunks = chunk_text(text)
        summaries = []

        for chunk in chunks:
            prompt = f"Summarize the following text into a {summary_length}: {chunk}"
            response = model.generate_content(prompt)
            summaries.append(response.text)

        final_prompt = f"Summarize the following summaries into a final {summary_length}: {' '.join(summaries)}"
        final_summary = model.generate_content(final_prompt)

        return final_summary.text

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# FastAPI endpoint
@app.post("/summarize-url/")
async def summarize_url(request: URLRequest):
    summary = scrape_and_summarize(request.url, request.summary_length)
    return {"summary": summary}

@app.get("/news/{sector}")
async def get_news_by_sector(sector: str):
    url = f"https://gnews.io/api/v4/search?q={sector}&lang=en&country=in&apikey={API_KEY}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    if "articles" in data and data["articles"]:
        return {"sector": sector, "articles": data["articles"]}
    else:
        return {"sector": sector, "message": "No articles found."}
