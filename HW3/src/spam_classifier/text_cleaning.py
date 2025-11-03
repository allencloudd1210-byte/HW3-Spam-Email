"""文字前處理工具。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<.*?>")
_NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
_MULTISPACE_PATTERN = re.compile(r"\s+")


def _ensure_nltk_resource(resource: str, download: str) -> None:
    try:
        nltk.data.find(resource)
    except LookupError:  # pragma: no cover - 只在第一次下載觸發
        nltk.download(download)


@dataclass
class TextCleaner:
    language: str = "english"
    remove_stopwords: bool = True
    lemmatize: bool = True

    def __post_init__(self) -> None:
        _ensure_nltk_resource("corpora/stopwords", "stopwords")
        _ensure_nltk_resource("corpora/wordnet", "wordnet")

        self._stopwords: set[str] = set(stopwords.words(self.language)) if self.remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if self.lemmatize else None

    def clean(self, text: str) -> str:
        text = text.lower()
        text = _URL_PATTERN.sub(" ", text)
        text = _HTML_PATTERN.sub(" ", text)
        text = _NON_ALPHA_PATTERN.sub(" ", text)
        tokens = text.split()

        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self._stopwords]

        if self.lemmatize and self._lemmatizer is not None:
            tokens = [self._lemmatizer.lemmatize(token) for token in tokens]

        text = " ".join(tokens)
        text = _MULTISPACE_PATTERN.sub(" ", text).strip()
        return text

    def batch_clean(self, texts: Iterable[str]) -> List[str]:
        return [self.clean(text) for text in texts]
