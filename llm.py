from dotenv import load_dotenv
load_dotenv()
import os
import ollama

from google import genai
from schema import Card

client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
LLM_MODEL = os.getenv("COGNI_LLM", "qwen2.5:3b-instruct")

SYSTEM = """You turn a piece of text into ONE knowledge card for a personal memory.

LANGUAGE RULE (most important): the card title and content must ALWAYS be written in English.
The input may be in Hindi, Marathi, or Hinglish (Hindi written in English letters). Translate its meaning into English.
Keep only names and proper nouns in their original form.

- Copy every name exactly as written, letter by letter. Never correct, translate, or change the spelling of a name, even if it looks like a typo.
- Do not start titles with words like "Remember" or "Note".

Other rules:
- Output exactly one card covering the whole text.
- The card must be understandable alone. Name the subject instead of using pronouns.
- Use only information in the text. Never add facts or advice.
- Stay close to the meaning. Never reverse or reinterpret it. Keep lists and specifics.
- If the source says who wrote or said this, name that person instead of "the author".
- Title is a short topic label. Content holds the details.

Example input:
"Bache titli ke peeche bhagte hain. Unke liye wahi sab kuch hai. Bade bhi bhaag rahe hain, bas titliyan badal gayi hain."
Example output:
{"title":"Chasing butterflies at every age","content":"Children run after butterflies, and for them catching one is everything. Adults are still running too, only the butterflies have changed.","tags":["childhood","chasing goals","adulthood"]}
"""

def extract_card(text: str, doc_title: str = ""):
    user = f"Source: {doc_title}\n\nText:\n{text}" if doc_title else text
    resp = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        format=Card.model_json_schema(),
        options={"temperature": 0},
    )
    return Card.model_validate_json(resp.message.content)



GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
gem = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

def generate(prompt: str) -> str:
    if gem:
        try:
            r = gem.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            if r.text:
                return r.text
        except Exception as e:
            print("Gemini failed, falling back to local:", e)
    r = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": 4096},
    )
    return r.message.content