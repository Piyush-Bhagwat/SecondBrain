from dotenv import load_dotenv
load_dotenv()

import sys
import os, re, asyncio, requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import html
import tempfile
from faster_whisper import WhisperModel
from chunker import words_to_paragraphs
import uuid
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler


_whisper = None
TOKEN = os.environ["TELEGRAM_TOKEN"]
ALLOWED = int(os.environ["ALLOWED_USER_ID"])
API = os.getenv("COGNI_API", "http://localhost:8000")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def transcribe(path: str):
    global _whisper
    if _whisper is None:   # loaded on first use so the bot starts fast
        _whisper = WhisperModel(
            os.getenv("WHISPER_MODEL", "small"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE", "int8"),
        )
    segments, info = _whisper.transcribe(path, vad_filter=True, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language

def confirm_kb(key: str):
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Save", callback_data=f"vs:{key}"),
        InlineKeyboardButton("Discard", callback_data=f"vd:{key}"),
    ]])


async def voice_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await allowed(update):
        return
    m = update.message

    # Handle Document (PDF) uploads
    if m.document:
        if m.document.mime_type == "application/pdf":
            await m.reply_text("Processing PDF...")
            try:
                tg_file = await m.document.get_file()
                # Use a temporary file for upload
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    await tg_file.download_to_drive(tmp.name)
                    tmp_path = tmp.name

                # We need to send this as a file to /save_pdf
                # Since the current 'call' function only does JSON POST,
                # we need a specialized function for file uploads.
                d = await asyncio.to_thread(call_pdf, tmp_path)
                await m.reply_text(f"Saved {d['saved']} cards from PDF:\n" + "\n".join(d["titles"]))
                os.remove(tmp_path)
                return
            except Exception as e:
                await m.reply_text(f"Failed to process PDF: {e}")
                return

    media = m.voice or m.audio
    if media and media.duration and media.duration > 600:
        await m.reply_text("Too long (max 10 minutes).")
        return
    await m.reply_text("Transcribing...")
    try:
        tg_file = await media.get_file()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "voice.ogg")
            await tg_file.download_to_drive(path)
            text, lang = await asyncio.to_thread(transcribe, path)
        if not text:
            await m.reply_text("Couldn't make out any speech.")
            return
        key = uuid.uuid4().hex[:8]
        context.bot_data.setdefault("pending", {})[key] = text
        preview = text if len(text) <= 3000 else text[:3000] + " ... (truncated here, full text will be saved)"
        await m.reply_text(f"Heard ({lang}):\n{preview}", reply_markup=confirm_kb(key))
    except Exception as e:
        import traceback; traceback.print_exc()
        await m.reply_text(f"Failed: {e}")

async def voice_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if q.from_user.id != ALLOWED:
        await q.answer("Not authorized", show_alert=True)
        return
    action, key = q.data.split(":", 1)
    pending = context.bot_data.setdefault("pending", {})
    text = pending.pop(key, None)
    if text is None:
        await q.answer("Expired. Send the voice note again.", show_alert=True)
        await q.edit_message_reply_markup(reply_markup=None)
        return
    await q.answer()
    preview = text if len(text) <= 3000 else text[:3000] + " ..."
    if action == "vd":
        await q.edit_message_text(f"{preview}\n\nDiscarded.")
        return
    await q.edit_message_text(f"{preview}\n\nSaving...")
    try:
        d = await asyncio.to_thread(call, "/save", {
            "text": words_to_paragraphs(text), "title": "", "source": "telegram-voice"})
        await q.edit_message_text(f"{preview}\n\nSaved {d['saved']} card(s):\n" + "\n".join(d["titles"]))
    except Exception as e:
        pending[key] = text   # keep it so Save can be tapped again
        await q.edit_message_text(f"{preview}\n\nFailed: {e}", reply_markup=confirm_kb(key))

def to_tg_html(text: str) -> str:
    text = html.escape(text, quote=False)                              # must be first
    for t in ("b", "i", "code"):                                # allow only safe tags back
        text = text.replace(f"&lt;{t}&gt;", f"<{t}>").replace(f"&lt;/{t}&gt;", f"</{t}>")
    text = text.replace("&lt;br&gt;", "\n")
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.M)              # remove --- lines
    text = re.sub(r"^\s*[\*\-]\s+", "• ", text, flags=re.M)            # bullets
    text = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", text, flags=re.M)  # headings -> bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)                # **bold**
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)  # *italic*
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)              # `code`
    return text

async def watch_sync(update: Update, proc):
    code = await proc.wait()
    if code != 0:   # crashed before ingest.py could report on its own
        await update.message.reply_text(f"Sync process exited with code {code}. Check ingest.log.")

async def sync_youtube_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await allowed(update):
        return
    env = os.environ.copy()
    for a in context.args:          # optional: /sync_youtube 10  -> look at the last 10 history items
        if a.isdigit():
            env["YT_MAX_HISTORY"] = a
    log = open(os.path.join(BASE_DIR, "ingest.log"), "a")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "ingest.py", cwd=BASE_DIR, env=env, stdout=log, stderr=log
    )
    await update.message.reply_text("YouTube sync started. I'll message you when it's done.")
    asyncio.create_task(watch_sync(update, proc))

def split_msg(text: str, limit: int = 4000):
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) + 2 > limit:
            parts.append(cur)
            cur = ""
        cur += ("\n\n" if cur else "") + para
    if cur:
        parts.append(cur)
    return parts

async def send(update: Update, text: str):
    for chunk in split_msg(text):
        try:
            await update.message.reply_text(to_tg_html(chunk), parse_mode="HTML")
        except Exception:
            await update.message.reply_text(chunk)   # plain text fallback

def call_pdf(path):
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "application/pdf")}
        r = requests.post(f"{API}/save_pdf", files=files, timeout=900)
        r.raise_for_status()
        return r.json()

def call_url(path, params):
    # /save_url in api.py uses query params for the 'url' argument
    query = "&".join([f"{k}={requests.utils.quote(v)}" for k, v in params.items()])
    r = requests.post(f"{API}{path}?{query}", timeout=900)
    r.raise_for_status()
    return r.json()

def call(path, payload, type = "post"):
    if type == "post":
        r = requests.post(f"{API}{path}", json=payload, timeout=900)
    else:
        r = requests.get(f"{API}{path}", params=payload, timeout=900)
    r.raise_for_status()
    return r.json()

async def allowed(update: Update) -> bool:
    uid = update.effective_user.id
    if uid != ALLOWED:
        await update.message.reply_text(f"Not authorized. Your Telegram ID is {uid}")
        return False
    return True

async def save_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("received save command:", context.args)

    if not await allowed(update):
        return
    # remove the "/save" prefix but keep the rest of the text (including line breaks)
    text = re.sub(r"^/save(@\w+)?\s*", "", update.message.text).strip()
    if not text:
        await update.message.reply_text("Usage: /save your text")
        return
    await update.message.reply_text("Saving...")
    try:
        d = await asyncio.to_thread(call, "/save", {"text": text, "title": ""})
        msg = f"Saved {d['saved']} card(s):\n" + "\n".join(d["titles"])
        if "connection" in d:
            msg += f"\n\n💡 {d['connection']}"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"Failed: {e}")

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE, mode="strict", q=None):
    if not await allowed(update):
        return
    if q is None:                             
        q = " ".join(context.args or []).strip()
    if not q:
        await update.message.reply_text("Usage: /ask your question")
        return
    print("received ask command:", q)
    await update.message.reply_text("Thinking...")
    try:
        d = await asyncio.to_thread(call, "/ask", {"question": q, "mode": mode})
        src = "\n".join(f"[{s['n']}] {s['title']} ({s['distance']})" for s in d["sources"])
        await send(update, d["answer"] + (f"\n\nSources:\n{src}" if src else ""))
    except Exception as e:
        await update.message.reply_text(f"Failed: {e}")

async def think_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("received think command:", context.args)

    await ask_cmd(update, context, mode="think")

async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await allowed(update):
        return

    text = update.message.text.strip()

    # Detect URL
    if re.match(r"^https?://\S+", text):
        await update.message.reply_text("Parsing URL...")
        try:
            # Using a dummy payload since /save_url takes a query param in our implementation
            # Wait, we should check if we want it as a POST with body or GET.
            # In api.py I used @app.post("/save_url") which defaults to query params for simple types.
            # Let's use a helper to call the API.
            d = await asyncio.to_thread(call_url, "/save_url", {"url": text})
            await update.message.reply_text(f"Saved {d['saved']} cards from link:\n" + "\n".join(d["titles"]))
        except Exception as e:
            await update.message.reply_text(f"Failed to parse URL: {e}")
        return

    await ask_cmd(update, context, mode="strict", q=text)

async def free_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ask_cmd(update, context, mode="free")

async def brief_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await allowed(update):
        return
    await update.message.reply_text("Gathering your daily briefing...")
    try:
        d = await asyncio.to_thread(call, "/brief", {}, type="get")
        await send(update, d["briefing"])
    except Exception as e:
        await update.message.reply_text(f"Failed to get briefing: {e}")

async def review_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await allowed(update):
        return
    try:
        d = await asyncio.to_thread(call, "/review", {}, type="get")
        if not d.get("card"):
            await update.message.reply_text("No cards found in memory.")
            return
        card = d["card"]
        msg = f"<b>Memory Jogger:</b>\n\n<i>{card['title']}</i>\n\n{card['content']}"
        await send(update, msg)
    except Exception as e:
        await update.message.reply_text(f"Failed to review: {e}")


app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("save", save_cmd))
app.add_handler(CommandHandler("sync_youtube", sync_youtube_cmd))
app.add_handler(CommandHandler("ask", ask_cmd))
app.add_handler(CommandHandler("free", free_cmd))
app.add_handler(CommandHandler("think", think_cmd))
app.add_handler(CommandHandler("brief", brief_cmd))
app.add_handler(CommandHandler("review", review_cmd))
app.add_handler(CallbackQueryHandler(voice_confirm, pattern=r"^v[sd]:"))
app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_msg))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))
app.run_polling()