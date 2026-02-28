def summarize_article():
    import json, os
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist

    INPUT = "/data/intermediate/article_clean.txt"
    OUTPUT = "/data/intermediate/summary.json"

    with open(INPUT, "r", encoding="utf-8") as f:
        article = f.read()

    sentences = sent_tokenize(article)
    tokens = word_tokenize(article.lower())
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    freq = FreqDist(filtered)

    scores: dict[str, float] = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq:
                scores[sent] = scores.get(sent, 0) + freq[word]

    ranked = [s for s in sorted(scores, key=scores.get, reverse=True)
              if not s.strip().endswith("?")]
    summary = " ".join(ranked[:5])

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({"summary": summary}, f, indent=2)

    print("Summary generated.")
