import re

from dotenv import load_dotenv
load_dotenv()

import json
import os
from llm import client, LLM_MODEL, generate
from store import collection, embed, CARDS_DIR
from datetime import datetime as dt

MAX_DISTANCE = 0.48   # strict mode: any card included
TOP_MAX = 0.40        # strict mode: best card must be at least this close
core_personality = open("personality.txt", encoding="utf-8").read().strip()

STRICT_PROMPT = """Answer the question using only the notes below.
- Write a clear answer in full sentences, 2-5 sentences or a short list if the question asks for a list.
- Never answer with just a note title or number.
- Do not add anything that isn't in the notes, and never reverse their meaning.
- If the notes don't contain the answer, say so.
- Cite note numbers like [1] after the facts they support.

CORE PERSONALITY: {core_personality}
NOTES:
{context}

QUESTION: {question}
META_DATA: {date: {date}}
"""

THINK_PROMPT = """You are Piyush's thinking partner. The notes below are background about him: treat them as facts about him, and use your own knowledge and reasoning to answer whatever he asks.

- Answer the actual request directly, whatever kind it is: advice, explanation, plan, comparison, brainstorm.
- Use the notes only where they're relevant. Don't force them in, and never invent facts about him that aren't in the notes.
- Be concise: as short as the request allows, usually under 150 words, with no intro or closing remarks. Go longer only if he asks for detail.
- Format for Telegram: short paragraphs, "• " for bullets, <b>bold</b> for key terms. Use only <b>, <i> and <code> tags. No markdown, no headings, no tables.
- Cite note numbers like [1] only where you rely on a note.
- always add '- by Gemini' at the end of your answer.

CORE PERSONALITY: {core_personality}

NOTES:
{context}

REQUEST: {question}
META_DATA: {date: {date}}
"""

FREE_PROMPT = """You are a helpful assistant running locally for Piyush. Answer his message using your own general knowledge.
Below are optional notes from his personal memory. Use them only if they clearly help with the message, and ignore them otherwise. Do not mention the notes unless you used them; if you did, cite them like [1].
Be concise. If you are not sure about something, say so instead of guessing. Plain text for Telegram: only <b>, <i> and <code> tags, no markdown, no headings, no tables.

CORE PERSONALITY: {core_personality}

NOTES:
{context}

MESSAGE: {question}"""




def load(card_id):
    return json.loads((CARDS_DIR / f"{card_id}.json").read_text())

def retrieve(queries, k_each=3, max_distance=0.55):
    seen, pairs = set(), []
    for q in queries:
        res = collection.query(query_embeddings=[embed(q, "search_query")], n_results=k_each)
        for i, d in zip(res["ids"][0], res["distances"][0]):
            if i not in seen and d <= max_distance:
                seen.add(i)
                pairs.append((i, d))
    return pairs


def answer_free(question: str):
    res = collection.query(query_embeddings=[embed(question, "search_query")], n_results=3)
    pairs = [(i, d) for i, d in zip(res["ids"][0], res["distances"][0]) if d <= 0.45]
    cards = [load(i) for i, _ in pairs]
    context = "\n\n".join(f"[{n+1}] {c['title']}\n{c['raw_text']}" for n, c in enumerate(cards)) or "(no relevant notes)"
    r = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": FREE_PROMPT.format(
            core_personality=core_personality, context=context, question=question)}],
        options={"temperature": 0.7, "num_ctx": 4096},
    )
    text = r.message.content
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    sources = [{"n": n + 1, "title": c["title"], "distance": round(d, 2)}
               for n, (c, (_, d)) in enumerate(zip(cards, pairs)) if (n + 1) in cited]
    return {"answer": text, "sources": sources}



def answer(question: str, k: int = 3, mode: str = "strict"):
    if len(question.split()) < 3:
        return {"answer": "That's too short to search. Ask something more specific.", "sources": []}
    
    if mode == "free":
        return answer_free(question)

    if mode == "think":
        pairs = retrieve([question, "my goals, skills and interests"], k_each=4)
        if not pairs:
            return {"answer": "Nothing relevant in memory.", "sources": []}
    else:
        res = collection.query(query_embeddings=[embed(question, "search_query")], n_results=k)
        pairs = [(i, d) for i, d in zip(res["ids"][0], res["distances"][0]) if d <= MAX_DISTANCE]
        if not pairs or min(d for _, d in pairs) > TOP_MAX:
            return {"answer": "Nothing relevant in memory.", "sources": []}

    cards = [load(i) for i, _ in pairs]
    context = "\n\n".join(f"[{n+1}] {c['title']}\n{c['raw_text']}" for n, c in enumerate(cards))
    sources = [{"n": n + 1, "title": c["title"], "distance": round(d, 2)}
               for n, (c, (_, d)) in enumerate(zip(cards, pairs))]

    if mode == "think":
        print("Using Gemini for thinking...")
        text = generate(THINK_PROMPT.format(core_personality=core_personality, date=dt.now(), context=context, question=question))   # Gemini
    else:
        print("Thinking...")
        r = client.chat(model=LLM_MODEL,
                        messages=[{"role": "user", "content": STRICT_PROMPT.format(core_personality=core_personality, date=dt.now(), context=context, question=question)}],
                        options={"temperature": 0, "num_ctx": 4096})                
        text = r.message.content

    return {"answer": text, "sources": sources}

if __name__ == "__main__":
    out = answer(input("Question: "))
    print(out["answer"])
    for s in out["sources"]:
        print(f"[{s['n']}] {s['title']} (distance {s['distance']})")