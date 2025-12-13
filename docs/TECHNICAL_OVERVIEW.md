Voynich transcription loader

- Location: `load_voynich_transcription.py`; data files live in `data/`.
- Input: `data/RF1b-e.txt` (Eva transcription).
- Core parse result: `pages` dict `page_id -> {"info": header string, "paragraphs": paragraphs}`.
  - Paragraph = list of line dicts; starts on markers `@P0` or `*P0`, ends on `=Pt` or when `<$>` appears.
  - Line dict = `{"id": "f1r.1", "marker": "@P0", "text": raw_text, "words": words}`.
  - Words split on `.` only; punctuation like `?` or `<->` is preserved; header `info` kept verbatim.
- Extra derived outputs (all UTF-8, indented, under `data/`):
  - `voynich_parsed.json`: full structured pages.
  - `voynich_parsed_by_paragraph.json`: page -> list of paragraphs, each paragraph as flat list of words.
  - `voynich_plain_text.json`: page -> string; paragraphs joined with `\n`, words joined with spaces.
  - `voynich_page_index.json`: ordered page list and page-to-number mapping (appearance order).
- Helpers in the script:
  - `parse_pages(source=...)` builds the structured pages.
  - `page_paragraph_words(..., cleaned=False)` returns paragraph word lists for a page id or assigned number; with `cleaned=True`, `<->` variants keep the first option and `<$>`, `<%>` are removed.
  - `page_plain_text(..., cleaned=False)` returns the plain-text string for that page (paragraphs separated by `\n`); honors `cleaned`.
  - `generate_outputs()` produces all JSON artifacts and returns `(pages, paragraphs_by_page, plain_texts, page_index, ordered_pages, page_to_number)` and also binds them at module scope.
  - Page resolution accepts page names or appearance-based numbers.
- Regenerate everything: `python load_voynich_transcription.py` (no main guard; variables remain accessible for IPython work).

Quick iteration over pages:
- After import, use the numbered accessor: `page_plain_text(pages, ordered_pages, i)` where `i` starts at 1 and increases. That returns a single string with paragraphs separated by `\n`.
- `ordered_pages[i-1]` gives the page id for that number; `page_to_number` maps page ids back to their numbers.

Cleaning and ambiguity resolution:
- `clean.clean_words(words, resolver=None, prob_thresh=0.5, gap_thresh=1.5)`  
  - `resolver`: mapping or callable that returns a candidate dict for a cleaned token (e.g. from `mapping_from_results`). If None, no replacements.  
  - `prob_thresh`: minimum `conf_abs` share (0..1 within that token’s candidate set) to allow replacement.  
  - `gap_thresh`: minimum ratio best/second-best (`conf_gap`) to allow replacement.  
  Tokens are cleaned first; replacements are applied only when both thresholds pass.
- `ambiguous_resolver.analyze_ambiguous(pages, currier="a", cleaned=False, freq_min=3, weights=(1.0,0.4,0.2))`  
  Builds Currier-specific models (word counts, char n-grams, bigrams), finds tokens with `?`, proposes up to 5 candidates per token (pattern-respecting). `freq_min` filters rare candidates; `weights` set contributions of word_prior, char_score, context_score. Use `currier="a"|"b"|"all"`; `cleaned=True` to analyze cleaned tokens.
- `ambiguous_resolver.write_results(path, results)` writes JSON; `mapping_from_results(results, prob_thresh=0.5, gap_thresh=1.5)` builds a resolver mapping keyed by cleaned token for use in `clean_words`.


TF-IDF / “most important words” per page (`tfidf_keyness.py`)
- On import: loads `voynich_plain_text.json` + `voynich_page_index.json` and builds TF-IDF over pages (no main guard). Exposes `plain_texts`, `ordered_pages`, `labels`, `docs`, `vectorizer`, `tfidf_matrix`, `vocab`, `top_terms`, `strong_top_terms`, `similar_pages`.
- `group_documents(plain_texts, ordered_pages, groups=None, currier="all", cleaned=False, resolver=None, prob_thresh=0.2, gap_thresh=1.5, exclude_hapax=False)`:
  - Default: one doc per page (optionally Currier A/B via `currier`, hapax removal via `exclude_hapax`, cleaning via `cleaned` + resolver).
  - Custom groups: pass dict or list of page ids/indices; values are concatenated into one doc.
- `build_tfidf(docs, **vectorizer_kwargs)`: wraps `TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, ...)`, returns `(vectorizer, X, vocab)`.
- `top_terms_for_doc(i, X, vocab, k=20, min_weight=0.0)` and `top_terms_by_label(...)`: per-doc ranked TF-IDF terms.
- `strong_terms(labels, X, vocab, min_weight=0.15, k=20)`: filters to docs with terms above a weight threshold (helps spot standout pages).
- `similarity_matrix(X)` and `top_similar(labels, X, n=5)`: cosine similarity across docs (useful for clustering/heatmaps of page similarity by key terms).

