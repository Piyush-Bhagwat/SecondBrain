# Cogni - AI Second Brain

Cogni is a personal knowledge management system (a "Second Brain") designed for rapid information capture and intelligent retrieval. It allows users to dump unstructured data from various sources and query it later using Retrieval-Augmented Generation (RAG), turning a collection of notes into a structured, linked knowledge graph.

IMPORTANT: CLAUDE WILL ALWAYS WORK ON 'CLAUDE' BRANCH

## Core Functionality

### 1. Information Ingestion (Capture)
Cogni provides multiple interfaces for frictionless capture:
- **Telegram Bot**: The primary interface for quick entry.
    - **Text**: Direct save via `/save` or plain text messages.
    - **Voice**: Voice notes are transcribed using `faster-whisper` (local) before being saved.
    - **PDFs**: Uploaded PDFs are parsed for text and converted into knowledge cards.
    - **URLs**: Web pages are scraped and cleaned using `trafilatura` to extract the main content.
    - **Images (OCR)**: Photos are processed via `easyocr` (local) to extract text, which is then synthesized into cards.
- **YouTube Sync**: `/sync_youtube` allows importing history and content directly from YouTube.

### 2. Knowledge Storage & Processing
Instead of saving long documents, Cogni breaks information down into **Cards**.
- **Chunking**: `chunker.py` implements recursive semantic chunking to ensure context is preserved across cards.
- **Synthesis**: `llm.py` uses local LLMs (via Ollama, e.g., `qwen2.5:3b-instruct`) to transform raw chunks into structured `Card` objects (Title, Content, Tags).
- **Vector Store**: Uses **ChromaDB** for high-performance similarity search.
- **Persistence**: Every card is saved as an individual JSON file in the `cards/` directory for durability and easy backup.
- **Knowledge Graph**: The system automatically detects relationships between new and existing cards, creating explicit links to build a non-linear network of knowledge.

### 3. Information Retrieval & Interaction
- **`/ask`**: Strict retrieval. Provides exact answers based solely on stored knowledge.
- **`/think`**: A reasoned approach. The LLM analyzes the retrieved knowledge to provide a more synthesized, thought-out response.
- **`/free`**: Conversational mode for general interaction.
- **`/brief`**: Generates a daily briefing based on recently stored information.
- **`/review`**: An Active Recall system implementing Spaced Repetition (SRS). It quizzes the user on their own knowledge and schedules reviews based on performance.
- **`/connect`**: Analyzes two different topics to synthesize unexpected connections and suggest new links for the knowledge graph.

## Technical Stack

- **Backend**: Python (FastAPI) providing the core API for the bot and potential future clients.
- **Bot**: `python-telegram-bot` for the user interface.
- **LLM/Embeddings**: Integrated via `llm.py` and `store.py`. Primarily utilizes Ollama for local, private synthesis and Gemini for high-end generation.
- **Transcription**: `faster-whisper` for high-accuracy local audio-to-text.
- **OCR**: `easyocr` for local image-to-text extraction.
- **Database**: ChromaDB (Vector Search) + JSON Files (Metadata/Links).

## Project Structure

- `bot.py`: Telegram bot logic, command handlers, and multimodal input routing.
- `api.py`: FastAPI REST API serving the bot and managing ingestion endpoints.
- `ask.py`: Logic for RAG-based querying, synthesis, and knowledge connection.
- `ingest.py`: Data import scripts (e.g., YouTube synchronization).
- `store.py`: The primary interface for ChromaDB and JSON card persistence.
- `llm.py`: LLM integration, prompting, and local OCR orchestration.
- `chunker.py`: Logic for splitting raw text into semantic cards.
- `schema.py`: Pydantic models defining the structure of Cards and Knowledge links.
- `cards/`: Flat-file storage for all saved knowledge JSONs.

## Future Ideas

1. **Multi-Modal "Capture" Expansion**
   - Browser Extension for quick clipping of web content.
