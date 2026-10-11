import re, json, logging, sys
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

import os
from llm import client, LLM_MODEL, generate
from store import collection, embed, CARDS_DIR
from datetime import datetime as dt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cogni-ask")

MAX_DISTANCE = 0.48   # strict mode: any card included
TOP_MAX = 0.40        # strict mode: best card must be at least this close
CORE_PERSONALITY = open("personality.txt", encoding="utf-8").read().strip()
core_personality = CORE_PERSONALITY


STRICT_PROMPT = """Answer the question using only the notes below.
- Write a clear answer in full sentences, 2-5 sentences or a short list if the question asks for a list.
- Never answer with just a note title or number.
- Do not add anything that isn't in the notes, and never reverse their meaning.
- If the notes don't contain the answer, say so.
- Cite note numbers like [1] after the facts they support.
- The answer may need combining facts from the notes (for example, a favorite place answers "where should we go"). Do that, and say which note you used. Only say the notes don't contain it if nothing in them is related.

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

@lru_cache(maxsize=128)
def load(card_id):
    # Cached JSON loading to avoid repeated disk reads for common cards
    try:
        return json.loads((CARDS_DIR / f"{card_id}.json").read_text())
    except FileNotFoundError:
        logger.error(f"Card not found: {card_id}")
        return None

def retrieve(queries, k_each=3, max_distance=0.55):
    seen, pairs = set(), []
    for q in queries:
        res = collection.query(query_embeddings=[embed(q, "search_query")], n_results=k_each)
        for i, d in zip(res["ids"][0], res["distances"][0]):
            if i not in seen and d <= max_distance:
                seen.add(i)
                pairs.append((i, d))

    # Knowledge Graph: Follow explicit links (1st degree)
    graph_links = set()
    for pid, _ in pairs:
        try:
            card = load(pid)
            for link in card.get("links", []):
                graph_links.add(link["id"])
        except:
            continue

    # Add linked cards to results
    for lid in graph_links:
        if lid not in seen:
            # We don't have a distance for these, so we use a default or a small value
            pairs.append((lid, 0.5))
            seen.add(lid)

    return pairs

def split_concepts(query: str):
    """Splits a connection query into two distinct concepts to search for."""
    prompt = f"""Extract the two main concepts being compared or connected in this query.
    Respond ONLY in JSON format: {{"concept_a": "...", "concept_b": "..."}}
    Query: {query}
    """
    res = generate(prompt, llm='local')
    try:
        cleaned = res.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned)
    except:
        return {"concept_a": query, "concept_b": ""}

def synthesize_connection(concept_a: str, cards_a: list, concept_b: str, cards_b: list):
    """Finds the non-obvious bridge between two sets of notes."""
    if not cards_a or not cards_b:
        return "I could only find information on one of these topics, so I can't draw a connection."

    ctx_a = "\\n\\n".join(f"[{c['id']}] {c['title']}: {c['content']}" for c in cards_a)
    ctx_b = "\\n\\n".join(f"[{c['id']}] {c['title']}: {c['content']}" for c in cards_b)

    prompt = f"""You are a synthesis engine. Find the non-obvious bridge between these two sets of notes.

    TOPIC A ({concept_a}):
    {ctx_a}

    TOPIC B ({concept_b}):
    {ctx_b}

    1. Analyze how Topic A informs, contradicts, or relates to Topic B.
    2. Provide a concise synthesis (1-2 paragraphs) that explains the conceptual link.
    3. Suggest 1-2 explicit links between specific cards if a strong relationship exists.

    Respond ONLY in this JSON format:
    {{
      "synthesis": "The detailed explanation...",
      "suggested_links": [
        {{"from": "card_id_a", "to": "card_id_b", "type": "related", "reason": "..."}}
      ]
    }}
    """
    res = generate(prompt, llm='local')
    try:
        cleaned = res.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {"synthesis": "Failed to synthesize a connection.", "suggested_links": []}

def generate_quiz_question(card: dict):
    """Generates a challenging question based on a card's content."""
    prompt = f"""You are a learning assistant. Based on the note below, create a challenging question that tests the core concept.
    Do NOT include the answer in the question.

    NOTE:
    Title: {card['title']}
    Content: {card['content']}

    Question:"""
    return generate(prompt, llm='local')

def evaluate_answer(question: str, user_answer: str, content: str):
    """Evaluates a user's answer against the original content."""
    prompt = f"""You are an expert evaluator. Compare the user's answer to the correct content.

    QUESTION: {question}
    USER ANSWER: {user_answer}
    CORRECT CONTENT: {content}

    Evaluate if the answer is 'Correct', 'Partial', or 'Wrong'.
    Provide a brief (1 sentence) explanation of why.

    Respond ONLY in JSON format: {{"verdict": "Correct" | "Partial" | "Wrong", "explanation": "..."}}
    """
    res = generate(prompt, llm='local')
    try:
        cleaned = res.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned)
    except:
        return {"verdict": "Wrong", "explanation": "Could not evaluate answer."}

def answer_free(question: str):

    logger.info(f"Processing free answer: {question[:50]}...")
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

    logger.info(f"Retrieving context for {mode} answer...")
    if mode == "think":
        pairs = retrieve([question, "my goals, skills and interests"], k_each=4)
        if not pairs:
            return {"answer": "Nothing relevant in memory.", "sources": []}
    else:
        res = collection.query(query_embeddings=[embed(question, "search_query")], n_results=k)
        pairs = [(i, d) for i, d in zip(res["ids"][0], res["distances"][0]) if d <= MAX_DISTANCE]
        if pairs:
            avg_distance = sum(d for _, d in pairs) / len(pairs)

            if avg_distance > 0.30:
                pairs = pairs[:3]
        if not pairs or min(d for _, d in pairs) > TOP_MAX:
            return {"answer": "Nothing relevant in memory.", "sources": []}

    cards = [load(i) for i, _ in pairs]
    context = "\n\n".join(f"[{n+1}] {c['title']}\n{c['raw_text']}" for n, c in enumerate(cards))
    sources = [{"n": n + 1, "title": c["title"], "distance": round(d, 2)}
               for n, (c, (_, d)) in enumerate(zip(cards, pairs))]

    if mode == "think":
        logger.info("Calling Gemini for Thinking mode...")
        text = generate(THINK_PROMPT.format(core_personality=core_personality, date=dt.now(), context=context, question=question), llm='gemini')
    else:
        logger.info("Calling local LLM for Strict mode...")
        r = client.chat(model=LLM_MODEL,
                        messages=[{"role": "user", "content": STRICT_PROMPT.format(core_personality=core_personality, date=dt.now(), context=context, question=question)}],
                        options={"temperature": 0, "num_ctx": 4096})
        text = r.message.content

    return {"answer": text, "sources": sources}

def summarize_briefing(cards: list):
    """Generates a summary of recent knowledge."""
    if not cards:
        return "No new knowledge stored in the last 24 hours."

    context = "\n\n".join(f"- {c['title']}: {c['content']}" for c in cards)
    prompt = f"""You are Piyush's personal memory assistant.
Below is a list of things Piyush learned or stored in the last 24 hours:

{context}

Please provide a cohesive, encouraging daily briefing.
- Summarize the main themes.
- Group related items.
- Keep it concise and formatted for Telegram (use <b>, <i>, <ul>).
- End with a thoughtful question or a "Connecting the dots" insight.
"""
    return generate(prompt, llm='local')

def find_connection(new_card, existing_cards):
    """Analyzes a new card against existing ones to find synthesis-level insights.
    Returns a structured link if a connection is found, otherwise None.
    """
    if not existing_cards:
        return None

    context = "\\n\\n".join(f"Note [{n+1}] {c['title']}\\n{c['content']}" for n, c in enumerate(existing_cards))
    prompt = f"""You are a synthesis engine. A new note was just added to the memory:
NEW NOTE: {new_card['title']} - {new_card['content']}

EXISTING NOTES:
{context}

Does the new note connect to, expand, or contradict any of the existing notes in a non-obvious way?
Look for conceptual bridges, thematic overlaps, or contradictions that provide a new insight.
If yes, respond ONLY in the following JSON format:
{{
  "link_id": "ID of the most relevant note",
  "type": "expands" | "contradicts" | "related",
  "reason": "One sentence explanation of the synthesis/insight"
}}
If no, respond with 'NONE'.
Be concise. Focus on synthesis over simple similarity.
"""
    res = generate(prompt, llm='local')
    if "NONE" in res.upper():
        return None
    try:
        # Clean the response in case the LLM adds markdown code blocks
        cleaned_res = res.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned_res)
    except Exception as e:
        logger.error(f"Failed to parse connection JSON: {e} | Response: {res}")
        return None

if __name__ == "__main__":
    out = answer(input("Question: "))
    print(out["answer"])
    for s in out["sources"]:
        print(f"[{s['n']}] {s['title']} (distance {s['distance']})")
