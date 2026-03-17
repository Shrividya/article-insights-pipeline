# PDF QA Chatbot with LangChain & OpenAI

A conversational question-answering system that loads PDF documents, indexes them into a vector store, and enables natural language querying using LangChain and OpenAI's GPT-3.5-turbo.

---

## Overview

This project uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about the contents of PDF files. It loads PDFs from a local directory, splits the text into chunks, stores them in a FAISS vector index, and uses a conversational chain with memory to answer follow-up questions contextually.

---

## Features

- Load and parse PDF files from a local directory
- Chunk documents intelligently using `RecursiveCharacterTextSplitter`
- Embed and index text using OpenAI Embeddings + FAISS
- Conversational memory to support multi-turn Q&A
- Powered by GPT-3.5-turbo via LangChain's `ConversationalRetrievalChain`

---

## Prerequisites

- Python 3.8+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

---

## Project Structure

```
project/
│
├── data/
│   └── pdf_files/
│       └── resume.pdf        # Place your PDF file here
│
├── .env               # OpenAI API key (not committed)
├── rag_resume.py                   # Main script
├── requirements.txt
└── README.md
```

---

## Usage

1. Place your PDF file(s) in `data/pdf_files/`.
2. Ensure your API key is set in .env file
3. Run the script:

```bash
python rag_resume.py
```

### Example Query & Output

```
Query : "What are Shri's top 5 skills?"

Output: 
Based on the resume, Shri's top 5 skills are:

1. Data Engineering – Apache Airflow 3, ETL/ELT pipelines, Snowflake, Databricks, PySpark
2. AI/LLM Engineering – RAG pipelines, LangChain, OpenAI API, NLP, Sentiment Analysis
3. Cloud & DevOps – AWS (Certified), GCP, Azure, Kubernetes, Docker, Terraform
4. Programming – Python, Java, SQL, FastAPI, GraphQL
5. Data Orchestration – Astronomer/Astro, Airflow DAGs, CI/CD integration
```

> **Note:** LLM output is non-deterministic. Results may vary slightly between runs even at `temperature=0`.

---

## Limitations

- Processes only the **first PDF** found in the directory. Extend `loader` logic to iterate over all files if needed.
- Output is **non-deterministic** — even with `temperature=0`, minor variations occur due to floating-point sampling and FAISS retrieval ordering.
- No persistent storage — the vector index is rebuilt on every run. For large PDFs, consider saving/loading the FAISS index with `vectorstore.save_local()` / `FAISS.load_local()`.
- Conversation memory is **in-memory only** and resets on each run.

