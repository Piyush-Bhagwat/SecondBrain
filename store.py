import json, uuid, datetime, pathlib, logging, sys, random
import chromadb
from llm import client, extract_card
from chunker import chunk_text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cogni-store")

CARDS_DIR = pathlib.Path("cards")
CARDS_DIR.mkdir(exist_ok=True)
EMBED_MODEL = "nomic-embed-text"

chroma = chromadb.PersistentClient(path="chroma_db")
collection = chroma.get_or_create_collection("cogni", metadata={"hnsw:space": "cosine"})

def embed_batch(texts: list[str], kind: str):
    """Batch embedding to reduce network roundtrips to Ollama."""
    if not texts:
        return []

    # Create prefixed prompts for all texts
    prompts = [f"{kind}: {t}" for t in texts]
    # Ollama embeddings API doesn't natively support batching in one call via the SDK
    # as a list, but we can simulate it or use the chat endpoint.
    # Actually, for nomic-embed-text, we have to call it per text unless using a custom provider.
    # But we can optimize by using a ThreadPool for the network calls.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda t: client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"], prompts))
    return results

def embed(text: str, kind: str):
    return embed_batch([text], kind)[0]

def index_cards_batch(cards: list[dict]):
    """Batch upsert into ChromaDB."""
    if not cards:
        return

    texts = [f"{c['title']}. {c['content']}" for c in cards]
    embeddings = embed_batch(texts, "search_document")

    collection.upsert(
        ids=[c["id"] for c in cards],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"]} for c in cards],
    )

def save(text: str, source: str = "manual", title: str = "", extra: dict = None):
    doc_id = uuid.uuid4().hex[:12]
    cards = []

    logger.info(f"Processing text from {source} into cards...")

    # 1. Chunking
    chunks = list(chunk_text(text))

    # 2. Extract cards from chunks (This is the slow part)
    for i, chunk in enumerate(chunks):
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
            "links": [], # Initialize empty links
        }
        # Sync write to disk
        (CARDS_DIR / f"{card['id']}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
        cards.append(card)

    # 3. Batch index into ChromaDB
    index_cards_batch(cards)

    # 4. Knowledge Graph: Automated Linking
    from ask import retrieve, find_connection
    for card in cards:
        # Find similar existing cards
        similar_pairs = retrieve([card["title"]], k_each=3)
        if similar_pairs:
            # Load the actual cards
            similar_cards = []
            for pid, dist in similar_pairs:
                try:
                    from ask import load
                    similar_cards.append(load(pid))
                except:
                    continue

            connection = find_connection(card, similar_cards)
            if connection:
                # connection is now a dict: {"link_id": ..., "type": ..., "reason": ...}
                card["links"].append({
                    "id": connection["link_id"],
                    "type": connection["type"],
                    "reason": connection["reason"]
                })
                # Update the JSON file with links
                (CARDS_DIR / f"{card['id']}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))

    logger.info(f"Successfully created {len(cards)} cards for doc {doc_id}")
    return cards

def reindex_all():
    logger.info("Reindexing all cards...")
    all_cards = []
    for f in CARDS_DIR.glob("*.json"):
        all_cards.append(json.loads(f.read_text()))
        if len(all_cards) >= 100:
            index_cards_batch(all_cards)
            all_cards = []
    index_cards_batch(all_cards)
    logger.info("Reindexing complete.")

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

def get_recent_cards(hours: int = 24):
    """Returns cards created within the last N hours."""
    logger.info(f"Fetching cards from the last {hours} hours...")
    now = datetime.datetime.now()
    recent = []
    for f in CARDS_DIR.glob("*.json"):
        try:
            c = json.loads(f.read_text())
            c_date = datetime.datetime.fromisoformat(c["date"])
            if (now - c_date).total_seconds() <= hours * 3600:
                recent.append(c)
        except Exception:
            continue
    return recent

def get_random_card():
    """Picks a random card from the library."""
    files = list(CARDS_DIR.glob("*.json"))
    if not files:
        return None
    f = random.choice(files)
    return json.loads(f.read_text())

