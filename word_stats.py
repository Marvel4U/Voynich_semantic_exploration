import json
import logging
import re
from pathlib import Path
from collections import Counter
from clean import clean_words

log = logging.getLogger(__name__)

folio_re = r"f(\d+)(r|v)(\d*)"

USE_CURRIER_FROM_TRANSCRIPT = True
data_dir = Path(__file__).parent / "data"
page_index_path = data_dir / "voynich_page_index.json"
_currier_by_page_cache = None


CURRIER_A_RANGES = [
    ("f1r", "f24v"),
    ("f31r", "f31v"),
    ("f88r", "f90v1"),
]
CURRIER_A_SINGLES = ["f25r", "f25v", "f32r", "f32v", "f33r", "f34r", "f34v", "f67r2", "f67v1", "f67v2", "f91v"]

CURRIER_B_RANGES = [
    ("f26r", "f30v"),
    ("f35r", "f39v"),
    ("f75r", "f84v"),
    ("f93r", "f96v"),
    ("f100r1", "f116r"),
]
CURRIER_B_SINGLES = ["f68r1", "f68r2", "f68v1", "f68v2"]

def currier_map_from_transcript():
    global _currier_by_page_cache
    if _currier_by_page_cache is None:
        try:
            idx = json.loads(page_index_path.read_text(encoding="utf-8"))
            _currier_by_page_cache = idx.get("currier_by_page", {}) or {}
        except Exception as exc:
            log.warning("Failed to load currier mapping from %s: %s", page_index_path, exc)
            _currier_by_page_cache = {}
    return _currier_by_page_cache

def _expand_range(start, end, ordered_pages=None):
    if ordered_pages:
        try:
            s_idx = ordered_pages.index(start)
            e_idx = ordered_pages.index(end)
        except ValueError:
            s_idx = e_idx = -1
        if s_idx != -1 and e_idx != -1:
            if s_idx > e_idx:
                s_idx, e_idx = e_idx, s_idx
            return ordered_pages[s_idx : e_idx + 1]
        log.warning("Range %s-%s not found in provided ordering; falling back", start, end)
    ms = re.fullmatch(folio_re, start)
    me = re.fullmatch(folio_re, end)
    if not ms or not me:
        return [start] if start == end else [start, end]
    sn, ss, sfx = int(ms.group(1)), ms.group(2), ms.group(3)
    en, es, efx = int(me.group(1)), me.group(2), me.group(3)
    if sn == en and ss == es and sfx and efx and sfx.isdigit() and efx.isdigit():
        return [f"f{sn}{ss}{i}" for i in range(int(sfx), int(efx) + 1)]
    if sfx or efx:
        log.warning("Unexpected suffix in range %s-%s", start, end)
    sides = ["r", "v"]
    idx = {s: i for i, s in enumerate(sides)}
    out = []
    for n in range(sn, en + 1):
        for side in sides:
            if n == sn and idx[side] < idx[ss]:
                continue
            if n == en and idx[side] > idx[es]:
                continue
            out.append(f"f{n}{side}")
    return out

def _collect_pages(range_specs, singles, ordered_pages=None):
    pages = []
    for start, end in range_specs:
        pages.extend(_expand_range(start, end, ordered_pages))
    pages.extend(singles)
    if ordered_pages:
        # preserve ordering where possible
        keep = set(pages)
        return [p for p in ordered_pages if p in keep]
    return sorted(set(pages))

def currier_page_filter(currier="all", ordered_pages=None, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    c = str(currier).lower()
    if c not in {"a", "b"}:
        return None
    if use_transcript:
        cmap = currier_map if currier_map is not None else currier_map_from_transcript()
        if cmap:
            keep = [pid for pid, cur in cmap.items() if str(cur).lower().startswith(c)]
            if ordered_pages:
                keep = [pid for pid in ordered_pages if pid in keep]
            return set(keep)
        log.warning("No currier mapping from transcript; falling back to static ranges")
    if c == "a":
        return set(_collect_pages(CURRIER_A_RANGES, CURRIER_A_SINGLES, ordered_pages))
    if c == "b":
        return set(_collect_pages(CURRIER_B_RANGES, CURRIER_B_SINGLES, ordered_pages))
    return None

def filter_pages(paragraphs_by_page, currier="all", use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    available = list(paragraphs_by_page.keys())
    keep = currier_page_filter(currier, ordered_pages=available, use_transcript=use_transcript, currier_map=currier_map)
    if not keep:
        return paragraphs_by_page
    return {pid: paras for pid, paras in paragraphs_by_page.items() if pid in keep}

def iter_paragraph_words(paragraphs_by_page, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    for _, paras in filter_pages(paragraphs_by_page, currier, use_transcript, currier_map).items():
        for para in paras:
            yield clean_words(para) if cleaned else para

def iter_words(paragraphs_by_page, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    for para in iter_paragraph_words(paragraphs_by_page, currier, cleaned, use_transcript, currier_map):
        for w in para:
            yield w

def word_counter(paragraphs_by_page, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    return Counter(iter_words(paragraphs_by_page, currier, cleaned, use_transcript, currier_map))

def vocab(counter):
    return set(counter.keys())

def word_length_counts(paragraphs_by_page, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    return Counter(len(w) for w in iter_words(paragraphs_by_page, currier, cleaned, use_transcript, currier_map))

def iter_word_bigrams(paragraphs_by_page, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    for para in iter_paragraph_words(paragraphs_by_page, currier, cleaned, use_transcript, currier_map):
        for a, b in zip(para, para[1:]):
            yield (a, b)

def word_bigram_counter(paragraphs_by_page, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    return Counter(iter_word_bigrams(paragraphs_by_page, currier, cleaned, use_transcript, currier_map))

def iter_char_ngrams(paragraphs_by_page, n=2, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    for w in iter_words(paragraphs_by_page, currier, cleaned, use_transcript, currier_map):
        for i in range(len(w) - n + 1):
            yield w[i:i + n]

def char_ngram_counter(paragraphs_by_page, n=2, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    return Counter(iter_char_ngrams(paragraphs_by_page, n, currier, cleaned, use_transcript, currier_map))

def type_token_ratio(paragraphs_by_page, currier="all", cleaned=False, use_transcript=USE_CURRIER_FROM_TRANSCRIPT, currier_map=None):
    wc = word_counter(paragraphs_by_page, currier, cleaned, use_transcript, currier_map)
    tokens = sum(wc.values())
    types = len(wc)
    return types / tokens if tokens else 0.0

def zipf_series(counter, top_n=200):
    freqs = [f for _, f in counter.most_common(top_n)]
    ranks = list(range(1, len(freqs) + 1))
    return ranks, freqs


def iter_word_edge_ngrams(paragraphs_by_page, n=2, currier="all", cleaned=False, position="start"):
    pos = position.lower()
    for w in iter_words(paragraphs_by_page, currier, cleaned):
        if len(w) < n:
            continue
        yield w[:n] if pos == "start" else w[-n:]


def word_edge_ngram_counter(paragraphs_by_page, n=2, currier="all", cleaned=False, position="start"):
    return Counter(iter_word_edge_ngrams(paragraphs_by_page, n, currier, cleaned, position))
