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
    sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, max_length=MAX_LENGTH, truncation=True)

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
            mlflow.log_metric(f"text_{i}_positive_prob", confidence_score)
            mlflow.log_param(f"text_{i}_predicted_label", label)

        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("device", "GPU" if torch.cuda.is_available() else "CPU")
        mlflow.log_metric("total_inference_time", end_time - start_time)

with DAG(
    dag_id="news_sentiment_dag",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False) as dag:


    run_pipeline = PythonOperator(
        task_id="run_sentiment_pipeline",
        python_callable=run_sentiment_pipeline)
    
dag