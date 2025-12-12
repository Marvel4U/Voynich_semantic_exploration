import json
import logging
import math
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable, Tuple

from clean import clean_word
from word_stats import currier_page_filter

log = logging.getLogger(__name__)


def _filter_pages(pages, currier="all"):
    keep = currier_page_filter(currier, ordered_pages=list(pages.keys()))
    if not keep:
        return pages
    return {pid: pdata for pid, pdata in pages.items() if pid in keep}


def _iter_paragraph_tokens(pages, currier="all", cleaned=False):
    for pid, pdata in _filter_pages(pages, currier).items():
        for p_idx, paragraph in enumerate(pdata["paragraphs"]):
            para_tokens = []
            for line_idx, line in enumerate(paragraph):
                words = line["words"]
                for w in words:
                    cw = clean_word(w) if cleaned else w
                    if cw:
                        para_tokens.append(cw)
            if para_tokens:
                yield pid, p_idx, para_tokens


def _core_vocab(word_counts: Counter, quantile=0.9):
    total = sum(word_counts.values())
    cutoff = total * quantile
    accum = 0
    vocab = []
    for w, c in word_counts.most_common():
        vocab.append(w)
        accum += c
        if accum >= cutoff:
            break
    return vocab


def _deletion_forms(word: str, k: int) -> Iterable[str]:
    if k == 1:
        for i in range(len(word)):
            yield word[:i] + word[i + 1 :]
    elif k == 2:
        for i in range(len(word) - 1):
            yield word[:i] + word[i + 2 :]


def _build_deletion_lexicon(vocab: Iterable[str]) -> Tuple[defaultdict, defaultdict]:
    del1 = defaultdict(set)
    del2 = defaultdict(set)
    for w in vocab:
        for f in _deletion_forms(w, 1):
            del1[f].add(w)
        for f in _deletion_forms(w, 2):
            del2[f].add(w)
    return del1, del2


def build_models(pages, currier="all", cleaned=False, core_quantile=0.9):
    word_counts = Counter()
    word_bigrams = Counter()
    char_counts = {2: Counter(), 3: Counter()}
    char_vocab = Counter()

    for _, _, tokens in _iter_paragraph_tokens(pages, currier=currier, cleaned=cleaned):
        word_counts.update(tokens)
        word_bigrams.update(zip(tokens, tokens[1:]))
        for w in tokens:
            char_vocab.update(w)
            for n in (2, 3):
                for i in range(len(w) - n + 1):
                    char_counts[n][w[i : i + n]] += 1

    core = _core_vocab(word_counts, quantile=core_quantile)
    del1, del2 = _build_deletion_lexicon(core)
    top_chars = [c for c, _ in char_vocab.most_common(6)]
    top_bigrams = [bg for bg, _ in char_counts[2].most_common(10)]

    totals = {n: sum(c.values()) for n, c in char_counts.items()}
    return {
        "word_counts": word_counts,
        "word_bigrams": word_bigrams,
        "char_counts": char_counts,
        "char_totals": totals,
        "char_vocab": char_vocab,
        "top_chars": top_chars,
        "top_bigrams": top_bigrams,
        "core_vocab": set(core),
        "del1": del1,
        "del2": del2,
    }


def find_ambiguous_tokens(pages, currier="all"):
    keep = currier_page_filter(currier, ordered_pages=list(pages.keys()))
    results = []
    for pid, pdata in pages.items():
        if keep and pid not in keep:
            continue
        for p_idx, paragraph in enumerate(pdata["paragraphs"]):
            flat = []
            for line_idx, line in enumerate(paragraph):
                for t_idx, w in enumerate(line["words"]):
                    cw = clean_word(w)
                    flat.append((w, cw, line_idx, line.get("id", ""), t_idx))
            for idx, (w, cw, line_idx, line_id, t_idx) in enumerate(flat):
                if not cw or "?" not in cw:
                    continue
                prev_tok = flat[idx - 1][1] if idx > 0 else None
                next_tok = flat[idx + 1][1] if idx + 1 < len(flat) else None
                results.append(
                    {
                        "page_id": pid,
                        "paragraph_idx": p_idx,
                        "line_idx": line_idx,
                        "line_id": line_id,
                        "token_idx": t_idx,
                        "raw": w,
                        "clean": cw,
                        "currier": currier,
                        "prev": prev_tok,
                        "next": next_tok,
                    }
                )
    return results


def _char_prob(word: str, models, k=0.1):
    parts = []
    for n in (2, 3):
        counts = models["char_counts"][n]
        total = models["char_totals"][n] + k * max(len(models["char_vocab"]), 1)
        if total == 0 or len(word) < n:
            continue
        probs = []
        for i in range(len(word) - n + 1):
            gram = word[i : i + n]
            c = counts.get(gram, 0)
            probs.append((c + k) / total)
        if probs:
            parts.append(sum(probs) / len(probs))
    if not parts:
        return 0.0
    return sum(parts) / len(parts)


def _context_log_prob(candidate, prev_tok, next_tok, models, k=0.1):
    if not prev_tok and not next_tok:
        return 0.0
    wb = models["word_bigrams"]
    vocab = len(models["word_counts"]) or 1
    total = sum(wb.values()) + k * vocab
    score = 0.0
    if prev_tok:
        c = wb.get((prev_tok, candidate), 0)
        score += (c + k) / total
    if next_tok:
        c = wb.get((candidate, next_tok), 0)
        score += (c + k) / total
    return score


def _combined_score(candidate, models, prev_tok=None, next_tok=None, weights=(1.0, 0.4, 0.2)):
    wc = models["word_counts"]
    max_freq = wc.most_common(1)[0][1] if wc else 1
    word_prior = wc.get(candidate, 0) / max_freq
    char_score = _char_prob(candidate, models)
    ctx_score = _context_log_prob(candidate, prev_tok, next_tok, models)
    a, b, c = weights
    return a * word_prior + b * char_score + c * ctx_score, {
        "word_prior": word_prior,
        "char_score": char_score,
        "context_score": ctx_score,
    }


def _candidate_variants(token: str, models):
    qcount = token.count("?")
    pattern = re.compile("^" + re.escape(token).replace("\\?", ".") + "$")
    # Pass 1: single-char fills only (exact length match)
    vocab_matches = {w for w in models["word_counts"].keys() if pattern.fullmatch(w)}
    if vocab_matches:
        return vocab_matches

    cands = set()
    # Pass 2: two-char fills using common bigrams if nothing matched
    if qcount and qcount <= 2 and models.get("top_bigrams"):
        fill_bigrams = models["top_bigrams"][:6]
        for repl in product(fill_bigrams, repeat=qcount):
            t = list(token)
            r_iter = iter(repl)
            out = []
            for ch in t:
                if ch == "?":
                    out.extend(next(r_iter))
                else:
                    out.append(ch)
            cands.add("".join(out))

    # Pass 3 backstop: single-char fills with common chars if still empty
    if not cands and qcount and qcount <= 2 and models["top_chars"]:
        fill_chars = models["top_chars"][:5]
        for repl in product(fill_chars, repeat=qcount):
            t = list(token)
            r_iter = iter(repl)
            for i, ch in enumerate(t):
                if ch == "?":
                    t[i] = next(r_iter)
            cands.add("".join(t))

    return cands


def propose_candidates(token_info, models, weights=(1.0, 0.4, 0.2)):
    token = token_info["clean"]
    prev_tok = token_info.get("prev")
    next_tok = token_info.get("next")
    candidates = []
    for cand in _candidate_variants(token, models):
        score, parts = _combined_score(cand, models, prev_tok=prev_tok, next_tok=next_tok, weights=weights)
        freq = models["word_counts"].get(cand, 0)
        candidates.append({"form": cand, "score": score, "freq": freq, **parts})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    if not candidates:
        return candidates
    max_s = candidates[0]["score"]
    exps = [math.exp(c["score"] - max_s) for c in candidates]
    denom = sum(exps) or 1.0
    for c, e in zip(candidates, exps):
        c["conf_abs"] = e / denom
    if len(candidates) > 1:
        gap = exps[0] / (exps[1] + 1e-9)
    else:
        gap = float("inf")
    for c in candidates:
        c["conf_gap"] = gap
    return candidates


def analyze_ambiguous(pages, currier="all", cleaned=False, weights=(1.0, 0.4, 0.2), freq_min=3):
    models = build_models(pages, currier=currier, cleaned=cleaned)
    targets = find_ambiguous_tokens(pages, currier=currier)
    results = []
    for t in targets:
        cands = propose_candidates(t, models, weights=weights)
        cands = [c for c in cands if c["freq"] >= freq_min]
        t["candidates"] = cands[:5]
        results.append(t)
    return results


def write_results(path, results):
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def mapping_from_results(results, prob_thresh=0.5, gap_thresh=1.5):
    mapping = {}
    for entry in results:
        if not entry.get("candidates"):
            continue
        best = entry["candidates"][0]
        if best["conf_abs"] >= prob_thresh and best["conf_gap"] >= gap_thresh:
            mapping[entry["clean"]] = best
    return mapping


if __name__ == "__main__":
    from load_voynich_transcription import pages

    results_a = analyze_ambiguous(pages, currier="a", cleaned=False, freq_min=3)
    results_b = analyze_ambiguous(pages, currier="b", cleaned=False, freq_min=3)
    write_results(Path("data/ambiguous_a.json"), results_a)
    write_results(Path("data/ambiguous_b.json"), results_b)