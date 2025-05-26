from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
import requests
import mlflow
import uuid

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

API_KEY = "c1607c63d3ae457898ed5ad03ddb36bb"
QUERY_LIST = ["technology", "finance", "health"]
PAGE_SIZE = 10
MODEL = "siebert/sentiment-roberta-large-english"
MAX_LENGTH = 128
MLFLOW_TRACKING_URI = "http://mlflow_server:5000"


def pull_news(**kwargs):
    all_news_items = []
    timeline = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    # pull three news for each topic
    for query in QUERY_LIST:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&"
            f"pageSize={PAGE_SIZE}&"
            f"language=en&"
            f"from={timeline}&"
            f"sortBy=publishedAt&"
            f"apiKey={API_KEY}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f"Error fetching news for {query}: {e}")

        articles = response.json().get("articles", [])
        # separate the details of each news and store them in a dictionary
        texts = [
            {
                "news_id": str(uuid.uuid4()),
                "text": f"{a['title']} {a.get('description', '')}".strip(),
                "topic": query,
            }
            for a in articles
        ]

        # combine all the dictionaries in a list
        all_news_items.extend(texts)

    # push the results to xcom
    kwargs["ti"].xcom_push(key="news_items", value=all_news_items)


def run_sentiment_pipeline(**kwargs):

    # pull the news using xcom and separate ids and texts
    news_items = kwargs["ti"].xcom_pull(key="news_items", task_ids="pull_news_task")
    ids = [text["news_id"] for text in news_items]
    texts = [text["text"] for text in news_items]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=MAX_LENGTH,
        truncation=True,
    )

    sentiment_results = sentiment_pipeline(texts)

    # track the results with mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        for item, sentiment_result in zip(news_items, sentiment_results):
            label = sentiment_result["label"]
            score = sentiment_result["score"]

            mlflow.log_metric(
                f"{item['topic']} - {item['news_id']} - Confidence", score
            )
            mlflow.log_param(f"{item['topic']} - {item['news_id']} - Label", label)

        mlflow.log_param("Model Name", MODEL)
        mlflow.log_param("Max Token Length", MAX_LENGTH)
        mlflow.log_param("Device", "GPU" if torch.cuda.is_available() else "CPU")

    kwargs["ti"].xcom_push(key="sentiment_results", value=sentiment_results)
    kwargs["ti"].xcom_push(key="news_ids", value=ids)


def save_results_to_postgres(**kwargs):

    # pull the results as well as news using xcom
    sentiment_results = kwargs["ti"].xcom_pull(
        key="sentiment_results", task_ids="run_sentiment_pipeline_task"
    )
    news_items = kwargs["ti"].xcom_pull(key="news_items", task_ids="pull_news_task")

    hook = PostgresHook(postgres_conn_id="news_records")
    conn = hook.get_conn()
    cursor = conn.cursor()

    # create the table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment_results (
            id SERIAL PRIMARY KEY,
            news_id TEXT,
            topic TEXT,
            news_text TEXT,
            predicted_label TEXT,
            prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    for item, sentiment_result in zip(news_items, sentiment_results):
        cursor.execute(
            """
            INSERT INTO news_sentiment_results (news_id, topic, news_text, predicted_label)
            VALUES (%s, %s, %s, %s);
            """,
            (item["news_id"], item["topic"], item["text"], sentiment_result["label"]),
        )

    conn.commit()
    cursor.close()
    conn.close()


with DAG(
    dag_id="news_sentiment_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
) as dag:

    pull_news_task = PythonOperator(task_id="pull_news_task", python_callable=pull_news)
    run_sentiment_pipeline_task = PythonOperator(
        task_id="run_sentiment_pipeline_task", python_callable=run_sentiment_pipeline
    )
    save_results_to_postgres_task = PythonOperator(
        task_id="save_results_to_postgres_task",
        python_callable=save_results_to_postgres,
    )

    pull_news_task >> run_sentiment_pipeline_task >> save_results_to_postgres_task
