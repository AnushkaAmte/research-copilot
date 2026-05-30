import re
import json
from pathlib import Path
import configparser
from transformers import AutoTokenizer

# ---------------- CONFIG ---------------- #

INPUT_DIR = Path("../../data/processed")
OUTPUT_DIR = Path("../../data/chunks")
config = configparser.ConfigParser()
config.read('constants.ini')


MODEL_NAME = config.get('CHUNK', 'MODEL_NAME')
CHUNK_SIZE = config.getint('CHUNK', 'CHUNK_SIZE')
OVERLAP = config.getint('CHUNK', 'OVERLAP')
MIN_SUBSECTION_TOKENS = config.getint('CHUNK', 'MIN_SUBSECTION_TOKENS')


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

def clean_markdown(text):

    # remove image placeholders
    text = re.sub(
        r"==>\s*picture\s*\[.*?\]\s*intentionally omitted\s*<==",
        "",
        text,
        flags=re.IGNORECASE
    )

    # remove references / acknowledgements
    stop_headers = [
        r"^##\s+references",
        r"^##\s+bibliography",
        r"^##\s+acknowledg",
    ]

    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:

        stop = False

        for pattern in stop_headers:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                stop = True
                break

        if stop:
            break

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    # merge broken sentence linebreaks
    text = re.sub(r"(?<!\n)\n(?!#|\n)", " ", text)

    # collapse spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ---------------- HEADER PARSING ---------------- #

def parse_sections(md_text):
    """
    Parse markdown headers into hierarchical sections.
    Supports:
    # title
    ## section
    ### subsection
    """

    pattern = r"^(#{1,3})\s+(.*)$"

    lines = md_text.split("\n")

    nodes = []

    current_section = None
    current_subsection = None

    buffer = []

    def flush_buffer():
        nonlocal buffer, current_section, current_subsection

        if not buffer:
            return

        content = "\n".join(buffer).strip()

        if content:
            nodes.append({
                "section": current_section,
                "subsection": current_subsection,
                "text": content
            })

        buffer = []

    for line in lines:

        match = re.match(pattern, line)

        if match:
            flush_buffer()

            level = len(match.group(1))
            title = match.group(2).strip()

            if level == 2:
                current_section = title
                current_subsection = None

            elif level == 3:
                current_subsection = title

        else:
            buffer.append(line)

    flush_buffer()

    return nodes


# ---------------- TOKEN HELPERS ---------------- #

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_tokens(text):

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + CHUNK_SIZE

        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(chunk_text)

        start += CHUNK_SIZE - OVERLAP

    return chunks


# ---------------- MERGING ---------------- #

def merge_small_subsections(nodes):

    merged = []

    buffer = None

    for node in nodes:

        token_count = count_tokens(node["text"])

        if token_count >= MIN_SUBSECTION_TOKENS:

            if buffer:
                merged.append(buffer)
                buffer = None

            merged.append(node)

        else:

            if buffer is None:

                buffer = node.copy()

            else:

                # only merge within same section
                if (
                    buffer["section"] ==
                    node["section"]
                ):

                    buffer["text"] += "\n\n" + node["text"]

                else:

                    merged.append(buffer)
                    buffer = node.copy()

    if buffer:
        merged.append(buffer)

    return merged


# ---------------- MAIN PIPELINE ---------------- #

for md_file in INPUT_DIR.glob("*.md"):

    print(f"Processing: {md_file.name}")

    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_text = clean_markdown(md_text)

    sections = parse_sections(md_text)

    sections = merge_small_subsections(sections)

    all_chunks = []

    chunk_id = 0

    for node in sections:

        text_chunks = chunk_tokens(node["text"])

        for chunk in text_chunks:

            all_chunks.append({
                "paper_id": md_file.stem,
                "section": node["section"],
                "subsection": node["subsection"],
                "chunk_id": chunk_id,
                "token_count": count_tokens(chunk),
                "text": chunk
            })

            chunk_id += 1

    output_path = OUTPUT_DIR / f"{md_file.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved {len(all_chunks)} chunks "
        f"for {md_file.name}"
    )

print("Done.")