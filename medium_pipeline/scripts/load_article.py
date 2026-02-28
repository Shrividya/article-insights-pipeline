def load_article():
    import os, sys

    INPUT = "/data/input/medium.txt"
    OUTPUT = "/data/intermediate/article_raw.txt"

    os.makedirs(os.path.dirname("/data/intermediate"), exist_ok=True)

    try:
        with open(INPUT, "r", encoding="utf-8") as f:
            article = f.read()
        print(f"Article loaded: {len(article)} characters")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT}", file=sys.stderr)
        sys.exit(1)

    if not article.strip():
        print("ERROR: Input file is empty.", file=sys.stderr)
        sys.exit(1)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(article)

    print(f"Raw article written to {OUTPUT}")