from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import requests
import torch
import mlflow
import time

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

API_KEY = "c1607c63d3ae457898ed5ad03ddb36bb"
QUERY = "technology"
PAGE_SIZE = 3
MODEL_NAME = "siebert/sentiment-roberta-large-english"
MAX_LENGTH = 128
MLFLOW_TRACKING_URI = "http://mlflow_server:5000"


def run_sentiment_pipeline():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        truncation=True,
    )

    url = f"https://newsapi.org/v2/everything?q={QUERY}&pageSize={PAGE_SIZE}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    articles = response.json().get("articles", [])
    texts = [f"{a['title']} {a.get('description', '')}".strip() for a in articles]

    with mlflow.start_run():
        start_time = time.time()
        results = sentiment(texts)
        end_time = time.time()

        for i, result in enumerate(results):
            label = result["label"]
            score = result["score"]
            confidence_score = 1 - score if label == "NEGATIVE" else score
            mlflow.log_metric(f"News {i+1} Positive Probability", confidence_score)
            mlflow.log_param(f"News {i+1} Predicted Label", label)

        mlflow.log_param("Model Name", MODEL_NAME)
        mlflow.log_param("Max Token Length", MAX_LENGTH)
        mlflow.log_param("Device", "GPU" if torch.cuda.is_available() else "CPU")
        mlflow.log_metric("Average Inference Time", (end_time - start_time) / len(texts))

        all_texts = "\n\n".join([f"{i+1}. News: {t}" for i, t in enumerate(texts)])
        mlflow.log_text(all_texts, "News Articles")
        


with DAG(
    dag_id="news_sentiment_dag",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    run_pipeline = PythonOperator(
        task_id="run_sentiment_pipeline", python_callable=run_sentiment_pipeline
    )
mlflow.log_text()