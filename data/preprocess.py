from typing import List
import re
import html

def _process_text(text: str):
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return text

def preprocess_texts(texts: List[str]):
    return [_process_text(t) for t in texts]