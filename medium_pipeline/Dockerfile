FROM apache/airflow:3.1.7

# Switch to root for system packages
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Airflow home to a writable directory
ENV AIRFLOW_HOME=/app/airflow
WORKDIR /app

# Create folders and fix permissions
RUN mkdir -p $AIRFLOW_HOME/logs \
    && mkdir -p /app/dag \
    && mkdir -p /app/plugins \
    && chown -R 50000:0 /app

# Copy NLP requirements
COPY --chown=50000:0 requirements.txt .

# Switch to airflow user for pip installs (matches base image expectations)
USER airflow

# Install Airflow providers and NLP dependencies
# NOTE: Do NOT reinstall apache-airflow itself — it's already in the base image
# and reinstalling without constraints causes dependency conflicts.
RUN pip install --no-cache-dir \
    apache-airflow-providers-fab \
    apache-airflow-providers-docker \
    "apache-airflow-providers-celery>=3.3.0" \
    -r requirements.txt

# Install torch separately using --extra-index-url (not --index-url) so that
# PyPI remains the primary index and only torch is fetched from the CPU wheel index.
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download transformer model
RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', \
         model='cardiffnlp/twitter-roberta-base-sentiment-latest', \
         truncation=True, max_length=512)"

# Pre-download NLTK data
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('stopwords', quiet=True)"

# Copy scripts
COPY --chown=50000:0 scripts/ /app/scripts/

# Set default Airflow home and log folder environment variables
ENV AIRFLOW__CORE__DAGS_FOLDER=/app/dag \
    AIRFLOW__CORE__PLUGINS_FOLDER=/app/plugins \
    AIRFLOW__CORE__BASE_LOG_FOLDER=$AIRFLOW_HOME/logs