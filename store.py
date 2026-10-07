import json, uuid, datetime, pathlib
import chromadb
from llm import client, extract_card
from chunker import chunk_text

CARDS_DIR = pathlib.Path("cards")
CARDS_DIR.mkdir(exist_ok=True)
EMBED_MODEL = "nomic-embed-text"

chroma = chromadb.PersistentClient(path="chroma_db")
collection = chroma.get_or_create_collection("cogni", metadata={"hnsw:space": "cosine"})

def embed(text: str, kind: str):
    # nomic-embed-text expects a prefix: "search_document" for stored items, "search_query" for questions
    r = client.embeddings(model=EMBED_MODEL, prompt=f"{kind}: {text}")
    return r["embedding"]

def index_card(card: dict):
    text = f"{card['title']}. {card['content']}"
    collection.upsert(
        ids=[card["id"]],
        embeddings=[embed(text, "search_document")],
        documents=[text],
        metadatas=[{"source": card["source"]}],
    )

def save(text: str, source: str = "manual", title: str = "", extra: dict = None):
    doc_id = uuid.uuid4().hex[:12]
    cards = []
    for i, chunk in enumerate(chunk_text(text)):
        c = extract_card(chunk, title)
        card = {
            "id": f"{doc_id}-{i}",
            "doc_id": doc_id,
            "doc_title": title,
            "chunk": i,
            "source": source,
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "meta": extra or {},
            **c.model_dump(),
            "raw_text": chunk,
        }
        (CARDS_DIR / f"{card['id']}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
        index_card(card)
        cards.append(card)
    return cards

def reindex_all():
    # rebuild Chroma from the JSON files (JSON is the source of truth)
    for f in CARDS_DIR.glob("*.json"):
        index_card(json.loads(f.read_text()))

def delete_card(card_id: str) -> bool:
    f = CARDS_DIR / f"{card_id}.json"
    if not f.exists():
        return False
    f.unlink()
    try:
        collection.delete(ids=[card_id])
    except Exception:
        pass
    return True