import os, sys

def load_article(input_path: str = "/data/input/medium.txt"):
    INPUT = input_path
    os.makedirs(os.path.dirname("/data/intermediate/"), exist_ok=True)
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
