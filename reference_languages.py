import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

import nltk
from nltk.corpus import europarl_raw

log = logging.getLogger(__name__)
base_path = Path(__file__).resolve().parent
data_path = base_path / "data"
cache_path = data_path / "reference_languages_stats.json"
data_path.mkdir(exist_ok=True)
max_tokens = 200_000
languages = ["english", "german", "french", "spanish"]


def normalize_token(tok):
    norm = unicodedata.normalize("NFD", tok)
    norm = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
    norm = "".join(ch for ch in norm if ch.isalpha())
    return norm.lower()


def ensure_corpus():
    try:
        europarl_raw.english.words()
    except LookupError:
        log.info("Downloading europarl_raw via nltk.download")
        nltk.download("europarl_raw")


def load_cache():
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            log.warning("Cache %s is corrupted; ignoring", cache_path)
    return {}


def counter_to_dict(counter):
    return {str(k): int(v) for k, v in counter.items()}


def dict_to_counter(data, cast_key=None):
    if not data:
        return Counter()
    if cast_key:
        return Counter({cast_key(k): int(v) for k, v in data.items()})
    return Counter({k: int(v) for k, v in data.items()})


def collect_tokens(lang):
    ensure_corpus()
    corpus = getattr(europarl_raw, lang)
    tokens = []
    for tok in corpus.words():
        nt = normalize_token(tok)
        if not nt:
            continue
        tokens.append(nt)
        if len(tokens) >= max_tokens:
            break
    return tokens


def char_ngrams(tokens, n):
    cnt = Counter()
    for w in tokens:
        if len(w) < n:
            continue
        for i in range(len(w) - n + 1):
            cnt[w[i : i + n]] += 1
    return cnt


def edge_ngrams(tokens, n, position):
    pos = position.lower()
    cnt = Counter()
    for w in tokens:
        if len(w) < n:
            continue
        cnt[w[:n] if pos == "start" else w[-n:]] += 1
    return cnt


def compute_stats(tokens):
    wc = Counter(tokens)
    wl_counts = Counter(len(w) for w in tokens)
    wb_counts = Counter(zip(tokens, tokens[1:]))
    cb_counts = char_ngrams(tokens, 2)
    ct_counts = char_ngrams(tokens, 3)
    start_bi = edge_ngrams(tokens, 2, "start")
    end_bi = edge_ngrams(tokens, 2, "end")
    start_tri = edge_ngrams(tokens, 3, "start")
    end_tri = edge_ngrams(tokens, 3, "end")
    return {
        "tokens": len(tokens),
        "types": len(wc),
        "wc": counter_to_dict(wc),
        "wl_counts": counter_to_dict(wl_counts),
        "wb_counts": counter_to_dict(Counter({" ".join(k): v for k, v in wb_counts.items()})),
        "cb_counts": counter_to_dict(cb_counts),
        "ct_counts": counter_to_dict(ct_counts),
        "start_bi": counter_to_dict(start_bi),
        "end_bi": counter_to_dict(end_bi),
        "start_tri": counter_to_dict(start_tri),
        "end_tri": counter_to_dict(end_tri),
    }


def save_cache(data):
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_language_stats():
    cache = load_cache()
    changed = False
    stats = cache if isinstance(cache, dict) else {}
    for lang in languages:
        if lang in stats and stats[lang].get("tokens"):
            continue
        tokens = collect_tokens(lang)
        stats[lang] = compute_stats(tokens) | {
            "language": lang,
            "max_tokens": max_tokens,
            "normalization": "lower + accent stripped + letters only",
        }
        changed = True
        log.info("Computed stats for %s (%d tokens, %d types)", lang, stats[lang]["tokens"], stats[lang]["types"])
    if changed:
        save_cache(stats)
    return stats


language_stats = build_language_stats()

