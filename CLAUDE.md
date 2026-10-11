# Cogni - AI Second Brain

IMPORTANT: CLAUDE WILL ALWAYS WORK ON 'CLAUDE' BRANCH

Cogni is a personal knowledge management system (a "Second Brain") that allows the user to dump information from various sources and query it later using RAG (Retrieval-Augmented Generation).

## Core Functionality

- **Information Ingestion**:
  - Telegram Bot interface for quick capture.
  - Supports text (`/save`), voice notes (transcribed via Whisper), PDFs, and URLs.
  - YouTube synchronization (`/sync_youtube`) to import history/content.
- **Knowledge Storage**:
  - Uses ChromaDB for vector storage of "cards" (chunks of information).
  - Information is broken down into cards for granular retrieval.
  - **Knowledge Graph**: Cards can have explicit links to other cards (`links` list in JSON), allowing for non-linear retrieval.
- **Information Retrieval**:
  - `/ask`: Strict retrieval to give exact answers based on stored knowledge.
  - `/think`: A more reasoned approach to querying knowledge.
  - `/free`: Free-form response handling.
  - `/brief`: Generates a daily briefing of stored info.
  - `/review`: A "memory jogger" to surface random cards for review.
  - `/connect`: Synthesizes connections between two different topics and suggests explicit links to build the knowledge graph.

## Technical Stack

- **Backend**: Python (FastAPI).
- **Bot**: `python-telegram-bot` for the interface.
- **LLM/Embeddings**: Integrated via `llm.py` and `store.py`.
- **Transcription**: `faster-whisper` for audio processing.
- **Database**: ChromaDB for vector search + JSON files for metadata and links.

## Project Structure

- `bot.py`: Telegram bot logic and command handlers.
- `api.py`: The REST API serving the bot and other clients.
- `ask.py`: Logic for querying the knowledge base and synthesizing connections.
- `ingest.py`: Logic for importing data (e.g., from YouTube).
- `store.py`: Interface for ChromaDB operations and JSON card persistence.
- `llm.py`: LLM integration and prompting.
- `chunker.py`: Logic for splitting text into cards.
- `cards/`: Directory containing JSON representations of saved knowledge.

## Future Ideas

1. **Multi-Modal "Capture" Expansion (Browser Ext & OCR)**
   - Browser Extension for quick clipping.
   - Image-to-Knowledge (OCR) for handwritten notes or screenshots.

2. **"Active Recall" & Spaced Repetition (The Learning Loop)**
   - Implement a Spaced Repetition System (SRS) to push stored info back into the user's active memory.

3. **Autonomous "Knowledge Curator" (The Nightly Process)**
   - Automated clustering of cards into topics and gap detection in the knowledge base.
