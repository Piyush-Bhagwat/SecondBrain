from dotenv import load_dotenv
load_dotenv()

import fcntl
import os, re, sys, json, pathlib, subprocess, requests
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from llm import client, LLM_MODEL

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

def notify(msg: str):
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": msg}, timeout=30)
    except Exception as e:
        print("Telegram notify failed:", e)

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

def is_relevant(title, channel, keywords):
    for k in keywords:                       # cheap exact match first
        if re.search(rf"\b{re.escape(k)}\b", title, re.I):
            return True
    prompt = (f"Topics I care about: {', '.join(keywords)}\n\n"
              f"Video title: {title}\nChannel: {channel}\n\n"
              "Is this video clearly about one of these topics? "
              "Answer relevant=true only if the title or channel clearly indicates it, otherwise false.")
    r = client.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                    format=Verdict.model_json_schema(), options={"temperature": 0})
    return Verdict.model_validate_json(r.message.content).relevant

def get_transcript(vid):
    try:
        t = YouTubeTranscriptApi().fetch(vid, languages=["en", "hi"])
        text = " ".join(s.text for s in t)
        return re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", text)).strip()
    except Exception as e:
        print(f"  no transcript for {vid}: {str(e)[:80]}")
        return None

def to_paragraphs(text, size=150):
    words = text.split()[:MAX_WORDS]
    return "\n\n".join(" ".join(words[i:i + size]) for i in range(0, len(words), size))

def save_video(v, text):
    r = requests.post(f"{API}/save", timeout=7200, json={
        "text": text,
        "title": f"YouTube: {v['title']}",
        "source": "youtube",
        "extra": {"video_id": v["id"],
                  "url": f"https://www.youtube.com/watch?v={v['id']}",
                  "channel": v.get("channel") or v.get("uploader")},
    })
    r.raise_for_status()
    return r.json()["saved"]

def main():
    keywords = load_keywords()
    stats = dict(checked=0, relevant=0, cards=0, no_transcript=0, too_long=0, failed=0)
    if not DRY:
        requests.get(API, timeout=10)        # fails early if the API isn't running
    seen = load_seen()

    for v in fetch_history(MAX_HISTORY):
        vid = v["id"]
        if vid in seen:
            continue
        stats["checked"] += 1
        title = v.get("title") or ""
        channel = v.get("channel") or v.get("uploader") or ""

        if not is_relevant(title, channel, keywords):
            with SKIPPED_LOG.open("a", encoding="utf-8") as f:
                f.write(f"{vid} | {title}\n")
            mark_seen(seen, vid)
            continue
        stats["relevant"] += 1

        dur = v.get("duration")
        if dur and dur > MAX_MINUTES * 60:
            stats["too_long"] += 1
            mark_seen(seen, vid)
            continue

        if DRY:
            print("WOULD INGEST:", title)
            continue

        text = get_transcript(vid)
        if not text:
            stats["no_transcript"] += 1     # not marked seen: retried next run
            continue
        try:
            stats["cards"] += save_video(v, to_paragraphs(text))
            mark_seen(seen, vid)
            print("saved:", title)
        except Exception as e:
            stats["failed"] += 1
            print("save failed:", title, e)

    msg = (f"YouTube sync done: {stats['checked']} new in history, {stats['relevant']} relevant, "
           f"{stats['cards']} cards saved, {stats['no_transcript']} without captions, "
           f"{stats['too_long']} too long, {stats['failed']} failed.")
    print(msg)
    if not DRY:
        notify(msg)

if __name__ == "__main__":
    try:
        fd = os.open("yt_sync.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        notify("A YouTube sync is already running. Skipped this one.")
        sys.exit(0)
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        if not DRY:
            notify(f"YouTube sync failed: {str(e)[:300]}")
    finally:
        os.close(fd)
        os.remove("yt_sync.lock")