import re, json, logging, sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from store import save, delete_card, CARDS_DIR
from ask import answer, load
import asyncio
import aiofiles
import time
import trafilatura
from PyPDF2 import PdfReader
import io

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cogni-api")

app = FastAPI()

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cogni-api")

app = FastAPI()

class SaveReq(BaseModel):
    text: str
    title: str = ""
    source: str = "web"
    extra: dict = {}

class AskReq(BaseModel):
    question: str
    mode: str = "strict"

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.post("/save")
async def save_note(req: SaveReq):
    start_time = time.time()
    logger.info(f"Saving note from {req.source} | Title: {req.title or 'N/A'}")

    # save() is still sync, running in threadpool via to_thread
    cards = await asyncio.to_thread(save, req.text, source=req.source, title=req.title, extra=req.extra)

    # Connection Discovery
    connection = None
    if cards:
        # Use the first card as the primary query for connection discovery
        from ask import retrieve, find_connection
        similar_pairs = retrieve([cards[0]["title"]], k_each=3)
        if similar_pairs:
            similar_cards = [await asyncio.to_thread(load, p[0]) for p in similar_pairs]
            connection = await asyncio.to_thread(find_connection, cards[0], similar_cards)

    elapsed = time.time() - start_time
    logger.info(f"Saved {len(cards)} cards in {elapsed:.2f}s")
    return {"saved": len(cards), "titles": [c["title"] for c in cards], "connection": connection}

@app.post("/ask")
async def ask_q(req: AskReq):
    start_time = time.time()
    logger.info(f"Ask request [{req.mode}]: {req.question[:50]}...")

    # answer() is still sync, running in threadpool
    res = await asyncio.to_thread(answer, req.question, mode=req.mode, k=5)

    elapsed = time.time() - start_time
    logger.info(f"Answered in {elapsed:.2f}s")
    return res

@app.get("/cards")
async def list_cards(limit: int = 50):
    logger.info(f"Listing cards (limit={limit})")
    cards = []
    # Optimized: read files asynchronously
    for f_path in CARDS_DIR.glob("*.json"):
        async with aiofiles.open(f_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
            c = json.loads(content)
            cards.append({k: c.get(k) for k in ("id", "title", "content", "date", "source")})

    cards.sort(key=lambda c: c["date"] or "", reverse=True)
    return {"total": len(cards), "cards": cards[:limit]}

@app.delete("/cards/{card_id}")
async def remove_card(card_id: str):
    if not re.fullmatch(r"[a-z0-9-]+", card_id):
        raise HTTPException(400, "bad id")

    logger.info(f"Removing card: {card_id}")
    if not await asyncio.to_thread(delete_card, card_id):
        raise HTTPException(404, "not found")

    return {"deleted": card_id}

@app.post("/save_url")
async def save_url(url: str):
    logger.info(f"Processing URL: {url}")
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise HTTPException(400, "Could not fetch URL content")

        text = trafilatura.extract(downloaded)
        if not text:
            raise HTTPException(400, "Could not extract meaningful text from URL")

        # Use existing save logic
        cards = await asyncio.to_thread(save, text, source="url", title=url)
        return {"saved": len(cards), "titles": [c["title"] for c in cards]}
    except Exception as e:
        logger.error(f"URL save failed: {e}")
        raise HTTPException(500, f"URL processing failed: {str(e)}")

@app.post("/save_pdf")
async def save_pdf(file: UploadFile = File(...)):
    logger.info(f"Processing PDF: {file.filename}")
    try:
        content = await file.read()
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)

        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"

        if not text.strip():
            raise HTTPException(400, "PDF contains no extractable text")

        cards = await asyncio.to_thread(save, text, source="pdf", title=file.filename)
        return {"saved": len(cards), "titles": [c["title"] for c in cards]}
    except Exception as e:
        logger.error(f"PDF save failed: {e}")
        raise HTTPException(500, f"PDF processing failed: {str(e)}")

@app.get("/brief")
async def get_briefing():
    from store import get_recent_cards
    from ask import summarize_briefing
    logger.info("Generating daily briefing...")
    cards = await asyncio.to_thread(get_recent_cards, hours=24)
    briefing = await asyncio.to_thread(summarize_briefing, cards)
    return {"briefing": briefing}

@app.get("/review")
async def get_review():
    from store import get_random_card
    logger.info("Picking random card for review...")
    card = await asyncio.to_thread(get_random_card)
    return {"card": card}
