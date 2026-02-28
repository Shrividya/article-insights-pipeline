def compute_readability():
    import json, os, re
    from nltk.tokenize import sent_tokenize, word_tokenize

    INPUT = "/data/intermediate/article_clean.txt"
    OUTPUT = "/data/intermediate/readability.json"

    def count_syllables(word: str) -> int:
        word = word.lower()
        if len(word) > 3 and word.endswith("e"):
            word = word[:-1]
        return max(1, len(re.findall(r"[aeiou]+", word)))

    with open(INPUT, "r", encoding="utf-8") as f:
        article = f.read()

    sentences = sent_tokenize(article)
    words = [w for w in word_tokenize(article) if w.isalpha()]
    total_syllables = sum(count_syllables(w) for w in words)

    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
    level = "Easy" if score >= 70 else "Moderate" if score >= 50 else "Difficult"
    result = {"score": round(score, 2), "level": level}

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Readability: {result}")
