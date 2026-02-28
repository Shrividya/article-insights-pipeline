import sys
import os

# Scripts are mounted via PVC at /app/scripts
sys.path.insert(0, '/app/scripts')

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta

from load_article import load_article
from analyse_sentiment import sentiment_analysis
from clean_article import clean_article
from compute_readability import compute_readability
from extract_entities import extract_entities
from extract_keywords import extract_keywords
from save_results import save_results
from summarize_article import summarize_article
from generate_snippets import generate_snippets

default_args = {
    'owner': 'nlp_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Paths backed by PVCs mounted at /app/data
INPUT_PATH  = '/app/data/input/medium.txt'
OUTPUT_PATH = '/app/data/output/nlp_results.json'

with DAG(
    dag_id='medium_article_nlp_pipeline',
    default_args=default_args,
    description='NLP pipeline for Medium article analysis',
    schedule='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['nlp', 'content', 'marketing'],
) as dag:

    t_load = PythonOperator(
        task_id='load_article',
        python_callable=load_article,
        op_kwargs={'input_path': INPUT_PATH},
        queue='default',
    )

    t_clean = PythonOperator(
        task_id='clean_article',
        python_callable=clean_article,
        queue='default',
    )

    t_readability = PythonOperator(
        task_id='compute_readability',
        python_callable=compute_readability,
        queue='default',
    )

    # ----- HEAVY NLP TASKS (nlp queue) -----
    t_keywords = PythonOperator(
        task_id='extract_keywords',
        python_callable=extract_keywords,
        queue='nlp',
    )

    t_summary = PythonOperator(
        task_id='summarize_article',
        python_callable=summarize_article,
        queue='nlp',
    )

    t_sentiment = PythonOperator(
        task_id='analyze_sentiment',
        python_callable=sentiment_analysis,
        queue='nlp',
    )

    t_entities = PythonOperator(
        task_id='extract_entities',
        python_callable=extract_entities,
        execution_timeout=timedelta(minutes=10),
        queue='nlp',
    )

    t_snippets = PythonOperator(
        task_id='generate_snippets',
        python_callable=generate_snippets,
        queue='nlp',
    )

    t_save = PythonOperator(
        task_id='save_results',
        python_callable=save_results,
        op_kwargs={'output_path': OUTPUT_PATH},
        queue='default',
    )

    # ----- TASK DEPENDENCIES -----
    t_load >> t_clean >> [t_keywords, t_summary, t_sentiment, t_entities, t_readability, t_snippets] >> t_save