import re, json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from store import save, delete_card, CARDS_DIR
from ask import answer

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
def home():
    return FileResponse("index.html")

@app.post("/save")
def save_note(req: SaveReq):
    cards = save(req.text, source=req.source, title=req.title, extra=req.extra)
    return {"saved": len(cards), "titles": [c["title"] for c in cards]}

@app.post("/ask")
def ask_q(req: AskReq):
    return answer(req.question, mode=req.mode, k=5)

@app.get("/cards")
def list_cards(limit: int = 50):
    cards = []
    for f in CARDS_DIR.glob("*.json"):
        c = json.loads(f.read_text())
        cards.append({k: c.get(k) for k in ("id", "title", "content", "date", "source")})
    cards.sort(key=lambda c: c["date"] or "", reverse=True)
    return {"total": len(cards), "cards": cards[:limit]}

@app.delete("/cards/{card_id}")
def remove_card(card_id: str):
    if not re.fullmatch(r"[a-z0-9-]+", card_id):
        raise HTTPException(400, "bad id")
    if not delete_card(card_id):
        raise HTTPException(404, "not found")
    return {"deleted": card_id}