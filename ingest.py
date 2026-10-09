import re, json, logging, sys, asyncio, httpx, os, pathlib, subprocess
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from llm import client, LLM_MODEL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cogni-ingest")

API = os.getenv("COGNI_API", "http://localhost:8000")
TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = int(os.environ["ALLOWED_USER_ID"])
BROWSER = os.getenv("YT_BROWSER", "firefox")
MAX_HISTORY = int(os.getenv("YT_MAX_HISTORY", "50"))
MAX_MINUTES = int(os.getenv("YT_MAX_MINUTES", "40"))
MAX_WORDS = int(os.getenv("YT_MAX_WORDS", "3000"))

STATE = pathlib.Path("yt_state.json")
SKIPPED_LOG = pathlib.Path("yt_skipped.log")
DRY = "--dry" in sys.argv

async def notify(msg: str):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                             json={"chat_id": CHAT_ID, "text": msg}, timeout=30)
    except Exception as e:
        logger.error(f"Telegram notify failed: {e}")

def load_keywords():
    lines = pathlib.Path("keywords.txt").read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]

def load_seen():
    return set(json.loads(STATE.read_text())["seen"]) if STATE.exists() else set()

def mark_seen(seen, vid):
    seen.add(vid)
    if not DRY:
        STATE.write_text(json.dumps({"seen": sorted(seen)}))

def fetch_history(n):
    cmd = [sys.executable, "-m", "yt_dlp", "--cookies-from-browser", BROWSER,
           "--flat-playlist", "--playlist-items", f"1-{n}", "--dump-json",
           "https://www.youtube.com/feed/history"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError("history fetch failed: " + out.stderr[-300:])
    return [json.loads(l) for l in out.stdout.splitlines() if l.strip()]

class Verdict(BaseModel):
    relevant: bool

async def is_relevant(title, channel, keywords):
    for k in keywords:
        if re.search(rf"\b{re.escape(k)}\b", title, re.I):
            return True
    prompt = (f"Topics I care about: {', '.join(keywords)}\n\n"
              f"Video title: {title}\nChannel: {channel}\n\n"
              "Is this video clearly about one of these topics? "
              "Answer relevant=true only if the title or channel clearly indicates it, otherwise false.")

    # Run blocking LLM call in thread
    def _call():
        r = client.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                        format=Verdict.model_json_schema(), options={"temperature": 0})
        return Verdict.model_validate_json(r.message.content).relevant

    return await asyncio.to_thread(_call)

async def get_transcript(vid):
    try:
        # Transcript API is sync
        t = await asyncio.to_thread(YouTubeTranscriptApi().fetch, vid, languages=["en", "hi"])
        text = " ".join(s.text for s in t)
        return re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", text)).strip()
    except Exception as e:
        logger.warning(f"No transcript for {vid}: {str(e)[:80]}")
        return None

def to_paragraphs(text, size=150):
    words = text.split()[:MAX_WORDS]
    return "\n\n".join(" ".join(words[i:i + size]) for i in range(0, len(words), size))

async def save_video(client, v, text):
    # Using httpx for async request
    # Increased timeout slightly and using a specific timeout config for the request
    print("Saving video:", v["title"])
    timeout = httpx.Timeout(120.0, connect=10.0, read=110.0)
    r = await client.post(f"{API}/save", timeout=timeout, json={
        "text": text,
        "title": f"YouTube: {v['title']}",
        "source": "youtube",
        "extra": {"video_id": v["id"],
                  "url": f"https://www.youtube.com/watch?v={v['id']}",
                  "channel": v.get("channel") or v.get("uploader")},
        "findConnection": False  # Disable connection discovery for YouTube videos
    })
    r.raise_for_status()
    print(f"Saved video: {v['title']} with status code {r.status_code}")
    # The API now returns {"status": "accepted", ...} instead of the card count
    return 1 # Treat as 1 successful operation for stats

async def process_video(v, keywords, seen, http_client, stats, processed_videos):
    vid = v["id"]
    if vid in seen:
        return

    title = v.get("title") or ""
    channel = v.get("channel") or v.get("uploader") or ""

    if not await is_relevant(title, channel, keywords):
        with SKIPPED_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{vid} | {title}\n")
        mark_seen(seen, vid)
        return

    stats["relevant"] += 1
    dur = v.get("duration")
    if dur and dur > MAX_MINUTES * 60:
        stats["too_long"] += 1
        mark_seen(seen, vid)
        return

    if DRY:
        logger.info(f"WOULD INGEST: {title}")
        return

    text = await get_transcript(vid)
    if not text:
        stats["no_transcript"] += 1
        return

    try:
        saved_count = await save_video(http_client, v, to_paragraphs(text))
        stats["cards"] += saved_count
        mark_seen(seen, vid)
        processed_videos.append(f"• {title} (https://www.youtube.com/watch?v={vid})")
        logger.info(f"Saved: {title} ({saved_count} cards)")
    except Exception as e:
        stats["failed"] += 1
        logger.error(f"Save failed for {title}: {repr(e)}")
        print(f"Save failed for {title}:")
        print(e)

async def main():
    keywords = load_keywords()
    stats = {"checked": 0, "relevant": 0, "cards": 0, "no_transcript": 0, "too_long": 0, "failed": 0}
    processed_videos = []

    if not DRY:
        try:
            async with httpx.AsyncClient() as client:
                await client.get(API, timeout=10)
        except Exception:
            logger.error("API is not running. Exiting.")
            return

    seen = load_seen()
    history = fetch_history(MAX_HISTORY)
    total_videos = len(history)
    logger.info(f"Fetched {total_videos} videos from history.")

    async with httpx.AsyncClient() as http_client:
        # Process videos concurrently in batches to avoid overloading local LLM/API
        # Batch size of 3 is safe for Ryzen 5 / 8GB RAM
        batch_size = 3
        for i in range(0, total_videos, batch_size):
            batch = history[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({i+1} to {min(i+batch_size, total_videos)} of {total_videos})")
            tasks = []
            for v in batch:
                stats["checked"] += 1
                tasks.append(process_video(v, keywords, seen, http_client, stats, processed_videos))

            await asyncio.gather(*tasks)

    # Construct final message with video list
    video_list_str = "\n".join(processed_videos)
    video_section = f"\n\nVideos processed:\n{video_list_str}" if processed_videos else ""

    msg = (f"YouTube sync done: {stats['checked']} new in history, {stats['relevant']} relevant, "
           f"{stats['cards']} cards saved, {stats['no_transcript']} without captions, "
           f"{stats['too_long']} too long, {stats['failed']} failed."
           f"{video_section}")
    logger.info(msg)
    if not DRY:
        await notify(msg)

if __name__ == "__main__":
    try:
        fd = os.open("yt_sync.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # notify is async now
        asyncio.run(notify("A YouTube sync is already running. Skipped this one."))
        sys.exit(0)
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("Critical sync error")
        asyncio.run(notify(f"YouTube sync failed: {str(e)[:300]}"))
    finally:
        os.close(fd)
        os.remove("yt_sync.lock")
