import json, logging, sys, random
import datetime
from typing import List, Dict, Any
from llm import generate
from store import load_all_cards, update_card_metadata_batch, write_card

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cogni-curator")

GAPS_FILE = "gaps.json"

def propose_topics(cards: List[Dict[str, Any]]) -> List[str]:
    """Analyze a sample of cards and propose high-level categories."""
    logger.info("Proposing topics based on knowledge base...")

    # Use a representative sample if too many cards
    sample_size = 50
    sample = random.sample(cards, min(len(cards), sample_size))

    context = "\n".join([f"Title: {c['title']}\nContent: {c['content']}" for c in sample])

    prompt = f"""Analyze the following list of knowledge cards. Propose a set of 5-10 distinct, high-level categories that best organize this information. Ensure categories are mutually exclusive and conceptually broad.

NOTES:
{context}

Respond ONLY in JSON format: {{"categories": ["Category A", "Category B", ...]}}"""

    res = generate(prompt, llm='local')
    try:
        cleaned = res.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned).get("categories", [])
    except Exception as e:
        logger.error(f"Topic proposal failed: {e}")
        return []

def assign_categories(cards: List[Dict[str, Any]], categories: List[str]) -> List[Dict[str, Any]]:
    """Assign each card to one of the proposed categories."""
    logger.info(f"Assigning cards to categories: {categories}")
    updates = []

    cat_list_str = ", ".join(categories)

    for card in cards:
        prompt = f"""Given the following card:
Title: {card['title']}
Content: {card['content']}

Which of these categories does it best fit into: {cat_list_str}?
Respond ONLY with the category name. If none fit, respond 'Uncategorized'."""

        category = generate(prompt, llm='local').strip()
        # Ensure we only use categories from the list or 'Uncategorized'
        if category not in categories:
            category = "Uncategorized"

        updates.append({"id": card["id"], "category": category})

    update_card_metadata_batch(updates)
    return updates

def detect_gaps(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify logical gaps for each category."""
    logger.info("Detecting knowledge gaps...")

    # Group cards by category
    by_cat = {}
    for c in cards:
        cat = c.get("category", "Uncategorized")
        by_cat.setdefault(cat, []).append(c)

    all_gaps = []

    for cat, cat_cards in by_cat.items():
        if cat == "Uncategorized" or len(cat_cards) < 2:
            continue

        context = "\n\n".join([f"Note: {c['title']}\n{c['content']}" for c in cat_cards])

        prompt = f"""You are a mental model architect. Here is the current knowledge base for the topic '{cat}':
{context}

Identify 2-3 logical gaps in this knowledge. What is missing to make this a complete, first-principles understanding of the topic?
Suggest specific questions or concepts that should be researched to fill these gaps.

Respond ONLY in JSON format: {{"category": "{cat}", "gaps": ["Gap 1 description", "Gap 2 description"]}}"""

        res = generate(prompt, llm='local')
        try:
            cleaned = res.strip().strip("```json").strip("```").strip()
            gap_data = json.loads(cleaned)
            gap_data["detected_at"] = datetime.datetime.now().isoformat()
            gap_data["resolved"] = False
            all_gaps.append(gap_data)
        except Exception as e:
            logger.error(f"Gap detection failed for {cat}: {e}")

    # Save gaps to file
    with open(GAPS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_gaps, f, indent=2)

    return all_gaps

def run_curation_cycle():
    """Main entry point for the nightly curation process."""
    logger.info("Starting Autonomous Curation Cycle...")

    cards = load_all_cards()
    if not cards:
        logger.info("No cards found to curate.")
        return

    # 1. Cluster
    categories = propose_topics(cards)
    if not categories:
        logger.warning("No categories proposed. Skipping assignment.")
    else:
        # 2. Categorize
        assign_categories(cards, categories)

    # 3. Gap Analysis
    # Reload cards to get updated categories
    updated_cards = load_all_cards()
    detect_gaps(updated_cards)

    logger.info("Curation Cycle Complete.")

if __name__ == "__main__":
    run_curation_cycle()
