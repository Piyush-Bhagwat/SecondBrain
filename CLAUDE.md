# Cogni - AI Second Brain

Cogni is a personal knowledge management system (a "Second Brain") that allows the user to dump information from various sources and query it later using RAG (Retrieval-Augmented Generation).

## Core Functionality
- **Information Ingestion**:
    - Telegram Bot interface for quick capture.
    - Supports text (`/save`), voice notes (transcribed via Whisper), PDFs, and URLs.
    - YouTube synchronization (`/sync_youtube`) to import history/content.
- **Knowledge Storage**:
    - Uses ChromaDB for vector storage of "cards" (chunks of information).
    - Information is broken down into cards for granular retrieval.
- **Information Retrieval**:
    - `/ask`: Strict retrieval to give exact answers based on stored knowledge.
    - `/think`: A more reasoned approach to querying knowledge.
    - `/free`: Free-form response handling.
    - `/brief`: Generates a daily briefing of stored info.
    - `/review`: A "memory jogger" to surface random cards for review.

## Technical Stack
- **Backend**: Python (FastAPI likely, based on `api.py`).
- **Bot**: `python-telegram-bot` for the interface.
- **LLM/Embeddings**: Integrated via `llm.py` and `store.py`.
- **Transcription**: `faster-whisper` for audio processing.
- **Database**: ChromaDB for vector search.

## Project Structure
- `bot.py`: Telegram bot logic and command handlers.
- `api.py`: The REST API serving the bot and other clients.
- `ask.py`: Logic for querying the knowledge base.
- `ingest.py`: Logic for importing data (e.g., from YouTube).
- `store.py`: Interface for ChromaDB operations.
- `llm.py`: LLM integration and prompting.
- `chunker.py`: Logic for splitting text into cards.
- `cards/`: Directory containing JSON representations of saved knowledge.
