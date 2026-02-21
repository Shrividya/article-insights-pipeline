from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import re
import json
import os

default_args = {
    'owner': 'nlp_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "~/airflow")
INPUT_PATH   = os.path.join(AIRFLOW_HOME, "data/input/medium.txt")
OUTPUT_PATH  = os.path.join(AIRFLOW_HOME, "data/output/nlp_results.json")


def load_article(**context):
    article = ""
    try:
        with open(INPUT_PATH, "r") as f:
            article = f.read()
        print(f"Article loaded: {len(article)} characters")
    except FileNotFoundError:
        print(f"ERROR: File not found at {INPUT_PATH}")
    context['ti'].xcom_push(key='article_text', value=article)


def extract_keywords(**context):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

    article = context['ti'].xcom_pull(key='article_text', task_ids='load_article')
    tokens = word_tokenize(article.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    freq = FreqDist(filtered)
    keywords = freq.most_common(10)
    result = [{"word": w, "count": c} for w, c in keywords]
    context['ti'].xcom_push(key='keywords', value=result)
    print(f"Keywords extracted: {result}")


def summarize_article(**context):
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

    article = context['ti'].xcom_pull(key='article_text', task_ids='load_article')
    sentences = sent_tokenize(article)
    tokens = word_tokenize(article.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    freq = FreqDist(filtered)

    scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq:
                scores[sent] = scores.get(sent, 0) + freq[word]

    ranked = sorted(scores, key=scores.get, reverse=True)
    summary = " ".join(ranked[:5])
    context['ti'].xcom_push(key='summary', value=summary)
    print("Summary generated.")


def analyze_sentiment(**context):
    from textblob import TextBlob

    article = context['ti'].xcom_pull(key='article_text', task_ids='load_article')
    blob = TextBlob(article)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    tone = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
    result = {"tone": tone, "polarity": round(polarity, 2), "subjectivity": round(subjectivity, 2)}
    context['ti'].xcom_push(key='sentiment', value=result)
    print(f"Sentiment: {result}")


def extract_entities(**context):
    import spacy

    article = context['ti'].xcom_pull(key='article_text', task_ids='load_article')

    # Debug: confirm article is being received via XCom
    print(f"Article length received: {len(article) if article else 0}")

    if not article:
        print("WARNING: No article text received from XCom")
        context['ti'].xcom_push(key='entities', value={})
        return

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article)

    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)

    # Deduplicate
    entities = {k: list(set(v)) for k, v in entities.items()}

    context['ti'].xcom_push(key='entities', value=entities)
    print(f"Entities extracted: {entities}")


def compute_readability(**context):
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    article = context['ti'].xcom_pull(key='article_text', task_ids='load_article')
    sentences = sent_tokenize(article)
    words = [w for w in word_tokenize(article) if w.isalpha()]

    def count_syllables(word):
        return max(1, len(re.findall(r'[aeiou]', word.lower())))

    total_syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
    level = "Easy" if score >= 70 else "Moderate" if score >= 50 else "Difficult"
    result = {"score": round(score, 2), "level": level}
    context['ti'].xcom_push(key='readability', value=result)
    print(f"Readability: {result}")


def generate_snippets(**context):
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    article = context['ti'].xcom_pull(key='article_text', task_ids='load_article')
    sentences = sent_tokenize(article)

    candidates = []
    for s in sentences:
        clean = s.replace("\n", " ").strip()
        clean = re.sub(r'\s+', ' ', clean)
        if len(clean.split()) > 12 and "?" not in clean:
            candidates.append(clean[:280])

    snippets = sorted(candidates, key=len, reverse=True)[:3]
    context['ti'].xcom_push(key='snippets', value=snippets)
    print(f"Snippets generated: {snippets}")


def save_results(**context):
    ti = context['ti']
    results = {
        "keywords":    ti.xcom_pull(key='keywords',    task_ids='extract_keywords'),
        "summary":     ti.xcom_pull(key='summary',     task_ids='summarize_article'),
        "sentiment":   ti.xcom_pull(key='sentiment',   task_ids='analyze_sentiment'),
        "entities":    ti.xcom_pull(key='entities',    task_ids='extract_entities'),
        "readability": ti.xcom_pull(key='readability', task_ids='compute_readability'),
        "snippets":    ti.xcom_pull(key='snippets',    task_ids='generate_snippets'),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_PATH}")
    print(json.dumps(results, indent=2))


with DAG(
    dag_id='medium_article_nlp_pipeline',
    default_args=default_args,
    description='NLP pipeline for Medium article analysis',
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['nlp', 'content', 'marketing'],
) as dag:

    t_load = PythonOperator(
        task_id='load_article',
        python_callable=load_article,
    )

    t_keywords = PythonOperator(
        task_id='extract_keywords',
        python_callable=extract_keywords,
    )

    t_summary = PythonOperator(
        task_id='summarize_article',
        python_callable=summarize_article,
    )

    t_sentiment = PythonOperator(
        task_id='analyze_sentiment',
        python_callable=analyze_sentiment,
    )

    t_entities = PythonOperator(
        task_id='extract_entities',
        python_callable=extract_entities,
        execution_timeout=timedelta(minutes=10),  # spaCy model loading needs extra time
    )

    t_readability = PythonOperator(
        task_id='compute_readability',
        python_callable=compute_readability,
    )

    t_snippets = PythonOperator(
        task_id='generate_snippets',
        python_callable=generate_snippets,
    )

    t_save = PythonOperator(
        task_id='save_results',
        python_callable=save_results,
    )

    t_load >> [t_keywords, t_summary, t_sentiment, t_entities, t_readability, t_snippets] >> t_save
