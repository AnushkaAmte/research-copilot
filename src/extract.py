import re
import json
from pathlib import Path
import configparser
from transformers import AutoTokenizer

# ---------------- CONFIG ---------------- #

INPUT_DIR = Path("../../data/processed")
OUTPUT_DIR = Path("../../data/chunks")
config = configparser.ConfigParser()
config.read('../constants.ini')


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
    r"^##\s+(?:\*\*)?references(?:\*\*)?",
    r"^##\s+(?:\*\*)?bibliography(?:\*\*)?",
    r"^##\s+(?:\*\*)?acknowledg(?:ement|ements)?(?:\*\*)?",
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

    lines = md_text.split("\n")

    nodes = []

    current_section = None
    current_subsection = None

    buffer = []

    KNOWN_HEADERS = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "method",
        "methods",
        "approach",
        "model architecture",
        "training",
        "experiments",
        "results",
        "evaluation",
        "discussion",
        "limitations",
        "conclusion",
        "references",
        "appendix"
    }

    def flush_buffer():
        nonlocal buffer

        content = "\n".join(buffer).strip()

        # don't create empty chunks and drop front matter sections with little content
        if (
            current_section is not None
            and content
            and len(content.split()) > 20
            ):

            nodes.append({
                "section": current_section,
                "subsection": current_subsection,
                "text": content
            })

        buffer = []

    for line in lines:

        line = line.strip()

        # markdown header
        match = re.match(
            r"^(#{1,6})\s+(.+)$",
            line
        )

        if match:

            level = len(match.group(1))
            title = match.group(2).strip()

            # remove markdown bold
            title = re.sub(r"\*\*", "", title)
            title = re.sub(r"\s+", " ", title)

            title_lower = title.lower()

            is_valid_header = False

            numbering_match = re.match(
                r"^(\d+(?:\.\d+)*)\s+(.+)$",
                title
            )

            # ---------- NUMBERED HEADERS ----------
            if numbering_match:

                numbering = numbering_match.group(1)
                depth = numbering.count(".")

                is_valid_header = True

                flush_buffer()

                # 3 Title
                if depth == 0:

                    current_section = title
                    current_subsection = None

                # 3.1 Subtitle
                elif depth == 1:

                    current_subsection = title

                # 3.2.1 Child subsection
                elif depth >= 2:

                    # keep under same subsection
                    # append title into body for context
                    buffer.append(f"\n{title}\n")

                continue

            # ---------- SEMANTIC HEADERS ----------
            for known in KNOWN_HEADERS:
                if known in title_lower:

                    is_valid_header = True

                    flush_buffer()

                    current_section = title
                    current_subsection = None
                    break

            if is_valid_header:
                continue

        # normal content
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


def chunk_text_semantically(
    text,
    chunk_size=500,
    overlap=75
):
    """
    Paragraph-aware token chunking.

    Strategy:
    1. Split by paragraphs
    2. Greedily pack paragraphs
    3. Sliding window fallback for huge paragraphs
    """

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []

    current_chunk = []
    current_tokens = 0

    for para in paragraphs:

        para_tokens = count_tokens(para)

        # -------------------------------
        # CASE 1: paragraph itself huge
        # -------------------------------
        if para_tokens > chunk_size:

            # flush existing chunk first
            if current_chunk:

                chunks.append(
                    "\n\n".join(current_chunk)
                )

                current_chunk = []
                current_tokens = 0

            tokens = tokenizer.encode(
                para,
                add_special_tokens=False
            )

            start = 0

            while start < len(tokens):

                end = start + chunk_size

                chunk_tokens = tokens[start:end]

                chunk_text = tokenizer.decode(
                    chunk_tokens
                )

                chunks.append(chunk_text)

                start += (
                    chunk_size - overlap
                )

            continue

        # -------------------------------
        # CASE 2: greedy paragraph packing
        # -------------------------------
        if (
            current_tokens + para_tokens
            <= chunk_size
        ):

            current_chunk.append(para)
            current_tokens += para_tokens

        else:

            chunks.append(
                "\n\n".join(current_chunk)
            )

            current_chunk = [para]
            current_tokens = para_tokens

    # final chunk
    if current_chunk:

        chunks.append(
            "\n\n".join(current_chunk)
        )
    if (
    len(chunks) > 1
    and count_tokens(chunks[-1]) < 150
    ):
        chunks[-2] += "\n\n" + chunks[-1]
        chunks.pop()
    return chunks

# ---------------- MERGING ---------------- #

def merge_small_subsections(nodes):

    merged = []

    buffer = None

    for node in nodes:

        token_count = count_tokens(node["text"])
        #print(f"Section: {node['section']} | Subsection: {node['subsection']} | Tokens: {token_count}")
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

                    buffer["text"] += (
                        f"\n\n{node['subsection']}\n\n"
                        + node["text"]
                    )

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

        text_chunks =  chunk_text_semantically(node["text"])

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