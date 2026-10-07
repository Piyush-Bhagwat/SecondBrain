from pydantic import BaseModel, Field
from typing import List, Literal

class Card(BaseModel):
    title: str = Field(description="Short topic label, 3-6 words, more specific than a single word")
    content: str = Field(description="2-4 sentences covering everything important in the text, naming subjects instead of using pronouns. Do not add words or claims not in the text.")
    tags: List[str] = Field(description="2-4 lowercase tags")