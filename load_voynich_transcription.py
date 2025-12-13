"""
Data shape:
- pages: dict[page_id] -> {"info": raw header string, "paragraphs": list[paragraph]}
- paragraph: list[line]; new paragraph starts on markers @P0 or *P0, ends on =Pt or when "<$>" appears in text
- line: {"id": "f1r.1", "marker": "@P0", "text": raw text, "words": list[str]}
- words: split on "." only; punctuation like "?" or "<->" is preserved inside words
Files: reads data/RF1b-e.txt, writes JSON outputs in data/
"""
from pathlib import Path
import json, logging, re
from clean import clean_words

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

data_dir = Path(__file__).parent / "data"
source_path = data_dir / "RF1b-er.txt"
parsed_path = data_dir / "voynich_parsed.json"
paragraph_path = data_dir / "voynich_parsed_by_paragraph.json"
plain_text_path = data_dir / "voynich_plain_text.json"
page_index_path = data_dir / "voynich_page_index.json"

start_markers = {"@P0", "*P0"}
end_markers = {"=Pt"}
page_re = re.compile(r"^<(?P<page_id>[^>]+)>\s+(?P<info><!.*>)")
line_re = re.compile(r"^<(?P<label>[^>]+)>\s+(?P<text>.*)$")
info_marker_re = re.compile(r"\$(?P<key>[A-Za-z])=(?P<value>[^\s>]+)")

def parse_page_meta(info):
    meta = {}
    for key, value in info_marker_re.findall(info or ""):
        meta[key.upper()] = value
    return meta

def parse_pages(source=source_path):
    pages = {}
    current_page = None
    current_info = None
    current_meta = None
    paragraphs = []
    current_paragraph = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        page_match = page_re.match(line)
        if page_match:
            if current_page:
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
                pages[current_page] = {"info": current_info, "meta": current_meta, "currier": current_meta.get("L") if current_meta else None, "paragraphs": paragraphs}
                paragraphs = []
            current_page = page_match.group("page_id")
            current_info = page_match.group("info").strip()
            current_meta = parse_page_meta(current_info)
            continue
        line_match = line_re.match(line)
        if not line_match:
            log.warning("Unparsed line: %s", line)
            continue
        label = line_match.group("label")
        text = line_match.group("text").strip()
        if "," in label:
            line_id, marker = label.split(",", 1)
            marker = marker.strip()
        else:
            line_id, marker = label, ""
        if marker in start_markers or not current_paragraph:
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
        words = [w for w in text.split(".") if w != ""]
        current_paragraph.append({"id": line_id, "marker": marker, "text": text, "words": words})
        if marker in end_markers or "<$>" in text:
            paragraphs.append(current_paragraph)
            current_paragraph = []
    if current_paragraph:
        paragraphs.append(current_paragraph)
    if current_page:
        pages[current_page] = {"info": current_info, "meta": current_meta, "currier": current_meta.get("L") if current_meta else None, "paragraphs": paragraphs}
    return pages

def write_json(data, dest):
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def build_page_order(pages):
    ordered = list(pages.keys())
    page_to_number = {pid: i + 1 for i, pid in enumerate(ordered)}
    return ordered, page_to_number

def build_currier_index(pages):
    return {pid: page.get("currier") for pid, page in pages.items() if page.get("currier")}

def resolve_page_id(page_or_number, ordered_pages):
    if isinstance(page_or_number, int):
        if page_or_number <= 0 or page_or_number > len(ordered_pages):
            raise ValueError(f"Page number {page_or_number} out of range 1..{len(ordered_pages)}")
        return ordered_pages[page_or_number - 1]
    return str(page_or_number)

def page_paragraph_words(pages, ordered_pages, page_or_number, cleaned=False):
    pid = resolve_page_id(page_or_number, ordered_pages)
    paragraphs = pages[pid]["paragraphs"]
    blocks = []
    for paragraph in paragraphs:
        words = [w for line in paragraph for w in line["words"]]
        words = clean_words(words) if cleaned else words
        blocks.append(words)
    return blocks

def page_plain_text(pages, ordered_pages, page_or_number, cleaned=False):
    blocks = page_paragraph_words(pages, ordered_pages, page_or_number, cleaned=cleaned)
    return "\n".join(" ".join(words) for words in blocks)

def generate_outputs():
    pages = parse_pages()
    ordered_pages, page_to_number = build_page_order(pages)
    currier_by_page = build_currier_index(pages)
    write_json(pages, parsed_path)
    paragraphs_by_page = {pid: page_paragraph_words(pages, ordered_pages, pid) for pid in ordered_pages}
    write_json(paragraphs_by_page, paragraph_path)
    plain_texts = {pid: page_plain_text(pages, ordered_pages, pid) for pid in ordered_pages}
    write_json(plain_texts, plain_text_path)
    page_index = {"ordered_pages": ordered_pages, "page_to_number": page_to_number, "currier_by_page": currier_by_page}
    write_json(page_index, page_index_path)
    log.info("Wrote %s pages into %s", len(pages), parsed_path.name)
    return pages, paragraphs_by_page, plain_texts, page_index, ordered_pages, page_to_number

pages, paragraphs_by_page, plain_texts, page_index, ordered_pages, page_to_number = generate_outputs()
currier_by_page = page_index.get("currier_by_page", {})