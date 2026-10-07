def chunk_text(text: str, max_words: int = 120):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur, count = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and count + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, count = [], 0
        cur.append(p)
        count += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def words_to_paragraphs(text: str, size: int = 150) -> str:
    words = text.split()
    return "\n\n".join(" ".join(words[i:i + size]) for i in range(0, len(words), size))

