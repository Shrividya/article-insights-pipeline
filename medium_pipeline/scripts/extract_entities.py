def extract_entities():
    import json, os, sys

    INPUT = "/data/intermediate/article_clean.txt"
    OUTPUT = "/data/intermediate/entities.json"

    VALID_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "NORP", "FAC", "PRODUCT"}

    with open(INPUT, "r", encoding="utf-8") as f:
        article = f.read()

    if not article.strip():
        print("WARNING: Empty article — writing empty entities.", file=sys.stderr)
        entities: dict = {}
    else:
        import spacy
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            print("WARNING: en_core_web_md not found, falling back to en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(article)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in VALID_LABELS:
                continue
            if ent.text == ent.text.lower() and " " not in ent.text:
                continue  # likely a misclassified common word
            entities.setdefault(ent.label_, []).append(ent.text)

        # Deduplicate each label's list
        entities = {k: list(set(v)) for k, v in entities.items()}

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(entities, f, indent=2)

    print(f"Entities extracted: {entities}")
