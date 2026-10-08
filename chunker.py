def chunk_text(text: str, max_words: int = 120, overlap: int = 20):
    """
    Recursive-style chunking with semantic overlap.
    Splits by paragraphs first, then by sentences/words if needed.
    """
    # 1. Split into paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for p in paras:
        p_words = p.split()
        p_count = len(p_words)

        # If a single paragraph is already too large, we must split it
        if p_count > max_words:
            # First, finish the current chunk if it exists
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                # Apply overlap: keep the last few words for the next chunk
                current_chunk_words = current_chunk_words[-overlap:] if overlap < len(current_chunk_words) else current_chunk_words
                current_word_count = len(current_chunk_words)

            # Split the large paragraph into sub-chunks
            for i in range(0, p_count, max_words - overlap):
                sub_chunk = " ".join(p_words[i : i + max_words])
                chunks.append(sub_chunk)

            # Reset for next paragraph
            current_chunk_words = []
            current_word_count = 0
        else:
            # Normal paragraph: check if it fits in current chunk
            if current_word_count + p_count > max_words:
                chunks.append(" ".join(current_chunk_words))
                # Overlap: take some words from the end of the previous chunk
                # For paragraph-based, we can't easily overlap mid-paragraph,
                # so we take the last paragraph if it's small, or just some words.
                overlap_words = current_chunk_words[-overlap:] if overlap < len(current_chunk_words) else current_chunk_words
                current_chunk_words = overlap_words + p_words
                current_word_count = len(current_chunk_words)
            else:
                current_chunk_words.extend(p_words)
                current_word_count += p_count

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks

def words_to_paragraphs(text: str, size: int = 150) -> str:
    """
    Convert raw text into paragraphs for better LLM processing.
    Now includes a small overlap to prevent context cutting.
    """
    words = text.split()
    paragraphs = []
    overlap = 15

    for i in range(0, len(words), size - overlap):
        chunk = words[i : i + size]
        paragraphs.append(" ".join(chunk))
        if i + size >= len(words):
            break

    return "\n\n".join(paragraphs)
