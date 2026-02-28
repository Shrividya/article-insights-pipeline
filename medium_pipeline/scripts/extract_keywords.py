def extract_keywords():
    import json, os
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist

    INPUT = "/data/intermediate/article_clean.txt"
    OUTPUT = "/data/intermediate/keywords.json"

    with open(INPUT, "r", encoding="utf-8") as f:
        article = f.read()

    tokens = word_tokenize(article.lower())
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    freq = FreqDist(filtered)
    keywords = [{"word": w, "count": c} for w, c in freq.most_common(10)]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(keywords, f, indent=2)

    print(f"Keywords extracted: {keywords}")