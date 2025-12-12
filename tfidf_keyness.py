import json, logging
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from clean import clean_words
from word_stats import currier_page_filter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

data_dir = Path(__file__).parent / "data"
plain_text_path = data_dir / "voynich_plain_text.json"
page_index_path = data_dir / "voynich_page_index.json"

def load_plain_texts(path=plain_text_path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_page_index(path=page_index_path):
    idx = json.loads(Path(path).read_text(encoding="utf-8"))
    return idx["ordered_pages"], idx["page_to_number"]

def resolve_page_id(page_or_number, ordered_pages):
    return ordered_pages[page_or_number - 1] if isinstance(page_or_number, int) else str(page_or_number)

def clean_text_block(text, resolver=None, prob_thresh=0.2, gap_thresh=1.5):
    toks = text.split()
    cleaned = clean_words(toks, resolver=resolver, prob_thresh=prob_thresh, gap_thresh=gap_thresh)
    return " ".join(cleaned)

def group_documents(plain_texts, ordered_pages, groups=None, currier="all", cleaned=False, resolver=None, prob_thresh=0.2, gap_thresh=1.5, exclude_hapax=False):
    keep = currier_page_filter(currier, ordered_pages=ordered_pages) if groups is None else None
    if groups is None:
        selected = [p for p in ordered_pages if (not keep) or (p in keep)]
        labels = selected
        docs = [plain_texts[pid] for pid in selected]
    else:
        items = groups.items() if isinstance(groups, dict) else enumerate(groups)
        labels, docs = [], []
        for label, group in items:
            grp = [group] if isinstance(group, (str, int)) else list(group)
            pids = [resolve_page_id(g, ordered_pages) for g in grp]
            labels.append(label)
            docs.append(" ".join(plain_texts[pid] for pid in pids))
    if cleaned:
        docs = [clean_text_block(d, resolver=resolver, prob_thresh=prob_thresh, gap_thresh=gap_thresh) for d in docs]
    if exclude_hapax:
        freq = Counter(tok for d in docs for tok in d.split())
        docs = [" ".join(tok for tok in d.split() if freq[tok] > 1) for d in docs]
    return labels, docs

def build_tfidf(docs, **vectorizer_kwargs):
    params = {"token_pattern": r"[^ ]+", "lowercase": False}
    params.update(vectorizer_kwargs)
    vec = TfidfVectorizer(**params)
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    return vec, X, vocab

def top_terms_for_doc(i, X, vocab, k=20, min_weight=0.0):
    row = X[i].toarray().ravel()
    nz = np.nonzero(row)[0]
    ordered = nz[np.argsort(row[nz])[::-1]]
    out = []
    for j in ordered:
        if row[j] <= min_weight:
            continue
        out.append((vocab[j], float(row[j])))
        if len(out) >= k:
            break
    return out

def top_terms_by_label(labels, X, vocab, k=20, min_weight=0.0):
    return {label: top_terms_for_doc(i, X, vocab, k=k, min_weight=min_weight) for i, label in enumerate(labels)}

def strong_terms(labels, X, vocab, min_weight=0.15, k=20):
    hits = {}
    for i, label in enumerate(labels):
        tt = top_terms_for_doc(i, X, vocab, k=k, min_weight=min_weight)
        if tt:
            hits[label] = tt
    return hits

def similarity_matrix(X):
    return (X @ X.T).toarray()

def top_similar(labels, X, n=5):
    sim = similarity_matrix(X)
    out = {}
    for i, label in enumerate(labels):
        order = np.argsort(sim[i])[::-1]
        neigh = [(labels[j], float(sim[i, j])) for j in order if j != i]
        out[label] = neigh[:n]
    return out

plain_texts = load_plain_texts()
ordered_pages, page_to_number = load_page_index()
labels, docs = group_documents(plain_texts, ordered_pages, currier="all", cleaned=False, exclude_hapax=False)
vectorizer, tfidf_matrix, vocab = build_tfidf(docs)
top_terms = top_terms_by_label(labels, tfidf_matrix, vocab, k=20)
strong_top_terms = strong_terms(labels, tfidf_matrix, vocab, min_weight=0.15, k=20)
similar_pages = top_similar(labels, tfidf_matrix, n=5)
log.info("Built TF-IDF for %d docs, vocab %d", tfidf_matrix.shape[0], tfidf_matrix.shape[1])

