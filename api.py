import re, json, logging, sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from store import save, delete_card, CARDS_DIR
from ask import answer, load
import asyncio
import aiofiles
import time

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
