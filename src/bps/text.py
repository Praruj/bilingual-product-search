"""Text normalisation and tokenisers for BM25."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

WORD = re.compile(r"[0-9a-z]+(?:[.'][0-9a-z]+)*")
JA_CHAR = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")


def normalize(s: str) -> str:
    """NFKC folds full-width letters and digits (ＡＢＣ１２３ -> abc123) and half-width katakana."""
    return unicodedata.normalize("NFKC", s or "").lower()


def has_japanese(s: str) -> bool:
    return bool(JA_CHAR.search(s or ""))


def whitespace_tokens(s: str) -> list[str]:
    """What a default English analyzer does. Japanese has no spaces, so a whole phrase becomes one token."""
    return normalize(s).split()


def word_tokens(s: str) -> list[str]:
    """English: lowercase alphanumeric words."""
    return WORD.findall(normalize(s))


def char_bigrams(s: str) -> list[str]:
    """Overlapping character bigrams inside each whitespace chunk: a dictionary-free tokenizer for Japanese."""
    out = []
    for chunk in normalize(s).split():
        if len(chunk) == 1:
            out.append(chunk)
        out.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return out


@lru_cache(maxsize=1)
def _tagger():
    import fugashi  # MeCab with the unidic-lite dictionary

    return fugashi.Tagger()


PUNCT = re.compile(r"^[\W_]+$")


def mecab_tokens(s: str) -> list[str]:
    """Japanese morphological analysis (MeCab, UniDic). Handles ASCII words in mixed text too."""
    return [w.surface for w in _tagger()(normalize(s)) if not PUNCT.match(w.surface)]


TOKENIZERS = {
    "whitespace": whitespace_tokens,
    "word": word_tokens,
    "char_bigram": char_bigrams,
    "mecab": mecab_tokens,
}
