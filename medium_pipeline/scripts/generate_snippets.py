def generate_snippets():
    import json, os, re
    from nltk.tokenize import sent_tokenize

    INPUT = "/data/intermediate/article_clean.txt"
    OUTPUT = "/data/intermediate/snippets.json"

    with open(INPUT, "r", encoding="utf-8") as f:
        article = f.read()

    sentences = sent_tokenize(article)
    candidates = []

    for s in sentences:
        clean = s.replace("\n", " ").strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean.split()) > 12 and "?" not in clean:
            candidates.append(clean)

    snippets = sorted(candidates, key=len, reverse=True)[:3]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(snippets, f, indent=2)

    print(f"Snippets generated: {snippets}")
