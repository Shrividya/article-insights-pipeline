def clean_article():
    import re, sys, os

    INPUT = "/data/intermediate/article_raw.txt"
    OUTPUT = "/data/intermediate/article_clean.txt"

    with open(INPUT, "r", encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        print("ERROR: Raw article is empty.", file=sys.stderr)
        sys.exit(1)

    lines = raw.splitlines()
    processed_lines: list[str] = []
    list_buffer: list[str] = []

    def flush_list(buf: list[str]) -> None:
        if not buf:
            return
        joined = ", ".join(item.strip(" \t\u2013\u2022-") for item in buf if item.strip())
        if joined:
            processed_lines.append(joined.rstrip(",") + ".")

    for line in lines:
        stripped = line.strip()
        is_list_item = (
                stripped
                and len(stripped) <= 60
                and stripped[-1] not in ".!?:\"'"
                and not stripped.startswith("#")
        )
        if is_list_item:
            list_buffer.append(stripped)
        else:
            flush_list(list_buffer)
            list_buffer = []
            processed_lines.append(line)

    flush_list(list_buffer)
    text = "\n".join(processed_lines)

    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse blank lines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # mid-para newlines → space
    text = re.sub(r'[ \t]+', ' ', text)  # collapse whitespace
    text = text.strip()

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Cleaned article: {len(text)} chars (was {len(raw)}). Written to {OUTPUT}")
