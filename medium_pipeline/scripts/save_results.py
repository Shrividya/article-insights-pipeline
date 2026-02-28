def save_results():
    import json, os, sys

    INTERMEDIATE = "/data/intermediate"
    OUTPUT = "/data/output/nlp_results.json"

    def load(filename: str):
        path = os.path.join(INTERMEDIATE, filename)
        with open(path, "r") as f:
            return json.load(f)

    try:
        results = {
            "keywords": load("keywords.json"),
            "summary": load("summary.json")["summary"],
            "sentiment": load("sentiment.json"),
            "entities": load("entities.json"),
            "readability": load("readability.json"),
            "snippets": load("snippets.json"),
        }
    except FileNotFoundError as e:
        print(f"ERROR: Missing intermediate file — {e}", file=sys.stderr)
        sys.exit(1)

    VALIDATION_RULES = {
        "keywords": lambda v: isinstance(v, list) and len(v) >= 5,
        "summary": lambda v: isinstance(v, str) and len(v.split()) >= 20,
        "sentiment": lambda v: isinstance(v, dict) and "tone" in v and "polarity" in v and "sentence_scores" in v,
        "entities": lambda v: isinstance(v, dict),
        "readability": lambda v: isinstance(v, dict) and 0 <= v.get("score", -1) <= 206.835,
        "snippets": lambda v: isinstance(v, list) and len(v) >= 1,
    }

    failures = []
    for field, rule in VALIDATION_RULES.items():
        value = results.get(field)
        if value is None:
            failures.append(f"  - '{field}': is None")
        elif not rule(value):
            failures.append(f"  - '{field}': failed quality check — got: {json.dumps(value)[:120]}")

    if failures:
        print("OUTPUT VALIDATION FAILED:", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        print("Aborting save to prevent writing incomplete results.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT}")
    print(json.dumps(results, indent=2))
