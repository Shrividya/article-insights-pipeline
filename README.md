# Article Insights Pipeline

An automated NLP pipeline for analysing Medium articles, orchestrated with **Apache Airflow 3** and deployable via **Docker Compose** or **Kubernetes**. Drop in a `.txt` article, trigger the DAG, and get keywords, summary, sentiment, named entities, readability scores, and social snippets — all validated and written to a single JSON file.

## How It Works

The DAG (`medium_article_nlp_pipeline`) runs on a daily schedule. It first loads and cleans the article, then fans out all heavy NLP tasks in parallel across two Celery queues, and finally aggregates, validates, and saves the results.

```
load_article
     │
  clean_article
     │
     ├──► extract_keywords      (nlp queue)
     ├──► summarize_article     (nlp queue)
     ├──► analyze_sentiment     (nlp queue)
     ├──► extract_entities      (nlp queue)
     ├──► compute_readability   (default queue)
     └──► generate_snippets     (nlp queue)
                    │
               save_results
```

Intermediate outputs are written to `/data/intermediate/` at each step. The final aggregated output is written to `/data/output/nlp_results.json` only after all validation checks pass.

---

## NLP Tasks

| Task | Script | What it does |
|------|--------|--------------|
| Load | `load_article.py` | Reads `medium.txt`, validates it is non-empty, writes raw text to intermediate storage |
| Clean | `clean_article.py` | Collapses bullet lists into prose, strips excess whitespace and mid-paragraph newlines |
| Keywords | `extract_keywords.py` | Top 10 words by frequency after stopword removal (NLTK) |
| Summary | `summarize_article.py` | Extractive summary — top 5 sentences ranked by word frequency score, questions excluded |
| Sentiment | `analyse_sentiment.py` | Sentence-level classification using `cardiffnlp/twitter-roberta-base-sentiment-latest` (Transformers); returns tone, polarity, and per-label averages |
| Entities | `extract_entities.py` | Named entity recognition via spaCy `en_core_web_md` (falls back to `en_core_web_sm`); extracts PERSON, ORG, GPE, LOC, EVENT, WORK_OF_ART, NORP, FAC, PRODUCT |
| Readability | `compute_readability.py` | Flesch-Kincaid score with Easy / Moderate / Difficult label |
| Snippets | `generate_snippets.py` | Top 3 longest sentences (>12 words, no questions) for use as social media captions |
| Save | `save_results.py` | Merges all intermediate JSON files, runs validation rules on every field, writes `nlp_results.json` — aborts if any check fails |

---

## Project Structure

```
article-insights-pipeline/
├── k8s/
│   ├── airflow/
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml
│   │   ├── airflow-db-migrate.yaml
│   │   ├── apiserver.yaml
│   │   ├── scheduler-deployment.yaml
│   │   ├── dag_processor.yaml
│   │   ├── worker-deployment.yaml
│   │   ├── nlp-worker-deployment.yaml
│   │   ├── triggerer.yaml
│   │   └── flower.yaml
│   ├── dag/
│   │   └── pipeline_dag.py          # Airflow DAG definition
│   ├── ingress/
│   │   └── ingress.yaml
│   ├── postgres/
│   │   ├── postgres-deployment.yaml
│   │   └── postgres-pvc.yaml
│   └── redis/
│       └── redis-deployment.yaml
├── scripts/
│   ├── load_article.py
│   ├── clean_article.py
│   ├── extract_keywords.py
│   ├── summarize_article.py
│   ├── analyse_sentiment.py
│   ├── extract_entities.py
│   ├── compute_readability.py
│   ├── generate_snippets.py
│   └── save_results.py
├── data/
│   ├── intermediate/                # Per-task JSON outputs (auto-created)
│   └── output/                      # Final nlp_results.json
├── logs/
├── plugins/
├── docker-compose.yaml
├── requirements.txt
└── test_nlp_pipeline.py
```

---

## Prerequisites

- Docker & Docker Compose (for local deployment)
- kubectl + a running Kubernetes cluster (for K8s deployment)
- Python 3.10+ (for running tests locally)

---

## Quick Start — Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/Shrividya/article-insights-pipeline.git
cd article-insights-pipeline
```

### 2. Set required environment variables

Generate a Fernet key and a secret key, then save them to a `.env` file in the project root:

```bash
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

echo "FERNET_KEY=$FERNET_KEY" >> .env
echo "SECRET_KEY=$SECRET_KEY" >> .env
```

### 3. Add your article

```bash
mkdir -p data/input
cp /path/to/your/article.txt data/input/medium.txt
```

### 4. Start the stack

```bash
docker compose up --build
```

This starts: **Postgres**, **Redis**, **Airflow API server** (port `8080`), **Scheduler**, **Celery Worker**, **DAG Processor**, **Triggerer**, and **Flower** (port `5555`).

### 5. Trigger the DAG

Open the Airflow UI at [http://localhost:8080](http://localhost:8080) (default credentials: `admin` / `admin`) and trigger `medium_article_nlp_pipeline`, or use the CLI:

```bash
docker compose exec airflow-scheduler airflow dags trigger medium_article_nlp_pipeline
```

### 6. View results

```bash
cat data/output/nlp_results.json
```

---

## Kubernetes Deployment

Apply the manifests in order:

```bash
# Namespace first
kubectl apply -f k8s/airflow/namespace.yaml

# Config and secrets
kubectl apply -f k8s/airflow/configmap.yaml
kubectl apply -f k8s/airflow/secrets.yaml

# Postgres and Redis
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/redis/

# Run DB migration job
kubectl apply -f k8s/airflow/airflow-db-migrate.yaml

# Airflow components
kubectl apply -f k8s/airflow/apiserver.yaml
kubectl apply -f k8s/airflow/scheduler-deployment.yaml
kubectl apply -f k8s/airflow/dag_processor.yaml
kubectl apply -f k8s/airflow/worker-deployment.yaml
kubectl apply -f k8s/airflow/nlp-worker-deployment.yaml
kubectl apply -f k8s/airflow/triggerer.yaml
kubectl apply -f k8s/airflow/flower.yaml

# Ingress
kubectl apply -f k8s/ingress/ingress.yaml
```

The pipeline uses two separate worker deployments: a **default worker** for lightweight tasks (load, clean, save) and a dedicated **NLP worker** (`nlp-worker-deployment.yaml`) for the CPU/GPU-intensive tasks routed to the `nlp` Celery queue.

---

## Output Format

```json
{
  "keywords": [
    { "word": "relationships", "count": 8 },
    { "word": "love", "count": 6 }
  ],
  "summary": "Emotional intimacy is a key foundation of a healthy relationship. Communication is essential for any relationship to thrive over time. ...",
  "sentiment": {
    "tone": "Positive",
    "polarity": 0.43,
    "sentence_scores": {
      "positive": 0.71,
      "neutral": 0.21,
      "negative": 0.08
    },
    "sentence_count": 12,
    "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
  },
  "entities": {
    "PERSON": ["Ada Lovelace"],
    "ORG": ["Medium"],
    "GPE": ["London"],
    "LOC": ["Silicon Valley"],
    "NORP": ["British"],
    "WORK_OF_ART": ["The Lean Startup"]
  },
  "readability": {
    "score": 62.4,
    "level": "Moderate"
  },
  "snippets": [
    "Emotional intimacy builds trust, provides emotional security, and relieves stress in any long-term relationship.",
    "Communication is essential for any relationship to thrive over time.",
    "Love can be unconditional, but relationships are a bit conditional."
  ]
}
```

### Output validation rules

`save_results.py` enforces these checks before writing — the DAG will fail rather than silently produce incomplete output:

| Field | Rule |
|-------|------|
| `keywords` | List with ≥ 5 entries |
| `summary` | String with ≥ 20 words |
| `sentiment` | Dict containing `tone`, `polarity`, and `sentence_scores` |
| `entities` | Dict (may be empty if no entities are detected) |
| `readability` | Dict with `score` in range `[0, 206.835]` |
| `snippets` | List with ≥ 1 entry |

---

## Running Tests

```bash
pip install pytest pytest-mock nltk
pytest test_nlp_pipeline.py -v
```

The test suite covers:

| Test Class | What is tested |
|-------|---------------|
| `TestCleanArticle` | Bullet list inlining, blank line collapsing, mid-paragraph newlines, markdown heading preservation |
| `TestCountSyllables` | Parametrised Flesch syllable counting including silent-e stripping and edge cases |
| `TestExtractKeywords` | Stopword removal, descending sort order, dict structure |
| `TestSummarizeArticle` | Question exclusion, sentences sourced from original article, top-N limit |
| `TestComputeReadability` | Score range, easy vs. hard text comparison, float type |
| `TestGenerateSnippets` | Minimum word count filter, question exclusion, longest-first ordering |
| `TestValidateResults` | All field validation rules including multi-failure reporting and empty entities |
| `TestLoadArticle` | File reading and missing-file handling via mock |
| `TestDagStructure` | DAG import and task dependency graph |

---

## Dependencies

Key packages (see `requirements.txt` for the full list):

| Package | Purpose |
|---------|---------|
| `apache-airflow` | Pipeline orchestration with CeleryExecutor |
| `nltk` | Tokenisation, stopwords, frequency distribution |
| `spacy` + `en_core_web_md` | Named entity recognition |
| `transformers` + `torch` | Transformer-based sentiment analysis |
| `textblob` | Supporting text utilities |
| `fastapi` / `uvicorn` | Airflow API server |
| `SQLAlchemy` + `alembic` | Metadata DB ORM and migrations |
| `pytest` + `pytest-mock` | Test suite |

---

## Architecture Notes

- **CeleryExecutor** with Redis as the broker and Postgres as the result backend
- **Two Celery queues**: `default` for lightweight I/O tasks and `nlp` for heavy model inference
- **Intermediate files** in `/data/intermediate/` decouple every task — each script reads its own input and writes its own output, so individual tasks can be retried without re-running the whole pipeline
- **Scripts mounted via PVC** in Kubernetes at `/app/scripts`, keeping the DAG image decoupled from business logic
- **DAG ID**: `medium_article_nlp_pipeline` | **Tags**: `nlp`, `content`, `marketing`
- **Schedule**: `@daily` with `catchup=False`
- **Retries**: 1 retry per task with a 5-minute delay
- **Entity task timeout**: 10 minutes (`extract_entities` is the slowest task)
