"""Lexical pre-processing for BM25.

Neural (dense) retrievers get the raw, lightly cleaned text instead: transformers were trained on
natural text and use their own sub-word tokenizer, so stemming or stop-word removal would hurt them.
"""

import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


class BM25Preprocessor:
    """lowercase -> tokenise -> (optional) stop-word removal -> (optional) Porter stemming."""

    def __init__(self, remove_stopwords=True, stem=True):
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self._stemmer = PorterStemmer() if stem else None
        self._cache = {}

    def __call__(self, text):
        tokens = _TOKEN.findall(text.lower())
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        if self.stem:
            out = []
            for t in tokens:
                s = self._cache.get(t)
                if s is None:
                    s = self._cache[t] = self._stemmer.stem(t)
                out.append(s)
            tokens = out
        return tokens

    def describe(self):
        parts = ["lowercase", "tokenise"]
        if self.remove_stopwords:
            parts.append("stop-words removed")
        if self.stem:
            parts.append("Porter stemming")
        return " + ".join(parts)
