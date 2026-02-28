def sentiment_analysis():
    import json, os
    from transformers import pipeline
    from nltk.tokenize import sent_tokenize

    INPUT = "/data/intermediate/article_clean.txt"
    OUTPUT = "/data/intermediate/sentiment.json"
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    with open(INPUT, "r", encoding="utf-8") as f:
        article = f.read()

    sentiment_pipeline = pipeline(
        task="text-classification",
        model=MODEL,
        truncation=True,
        max_length=512,
    )

    sentences = sent_tokenize(article)
    label_scores: dict[str, list[float]] = {"positive": [], "neutral": [], "negative": []}

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        output = sentiment_pipeline(sent)[0]
        label = output["label"].lower()
        if label in label_scores:
            label_scores[label].append(output["score"])

    averages = {
        label: (round(sum(scores) / len(scores), 4) if scores else 0.0)
        for label, scores in label_scores.items()
    }

    overall_tone = max(averages, key=averages.get).capitalize()
    polarity = round(averages["positive"] - averages["negative"], 4)

    result = {
        "tone": overall_tone,
        "polarity": polarity,
        "sentence_scores": averages,
        "sentence_count": len(sentences),
        "model": MODEL,
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Sentiment: {result}")
