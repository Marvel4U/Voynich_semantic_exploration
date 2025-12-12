def clean_word(word: str) -> str:
    base = word.split("<->", 1)[0]
    base = base.replace("<$>", "").replace("<%>", "")
    return base

def clean_words(words, resolver=None, prob_thresh=0.2, gap_thresh=1.5):
    out = []
    for w in (clean_word(w) for w in words):
        if not w:
            continue
        if resolver:
            candidate = resolver(w) if callable(resolver) else resolver.get(w)
            if candidate and candidate.get("conf_abs", 0) >= prob_thresh and candidate.get("conf_gap", 0) >= gap_thresh:
                w = candidate["form"]
        out.append(w)
    return out

