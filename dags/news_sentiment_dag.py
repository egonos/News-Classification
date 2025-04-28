from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator

from datetime import datetime
import requests
import mlflow
import time
import uuid

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

API_KEY = "c1607c63d3ae457898ed5ad03ddb36bb"
QUERY = "technology"
PAGE_SIZE = 3
MODEL = "siebert/sentiment-roberta-large-english"
MAX_LENGTH = 128
MLFLOW_TRACKING_URI = "http://mlflow_server:5000"


def pull_news(query=QUERY, page_size=PAGE_SIZE, **kwargs):

    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&language=en&sortBy=publishedAt&apiKey={API_KEY}"

    # fetch the data and check the presence of an error
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise Exception(f"Error fetching news: {e}")

    # get the data
    articles = response.json().get("articles", [])
    texts = [
        {
            "news_id": str(uuid.uuid4()),
            "text": f"{a['title']} {a.get('description', '')}".strip(),
        }
        for a in articles
    ]

    # push the data using xcom
    kwargs["ti"].xcom_push(key="news_items", value=texts)


def run_sentiment_pipeline(**kwargs):

    # pull the news texts using xcom
    news_items = kwargs["ti"].xcom_pull(key="news_items", task_ids="pull_news_task")
    ids = [text["news_id"] for text in news_items]
    texts = [text["text"] for text in news_items]

    # build the classification pipeline
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

    # run the classification pipeline and log the results in MLflow
    start_time = time.time()
    sentiment_results = sentiment_pipeline(texts)
    end_time = time.time()

    # track the sentiment results and news ids in MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        for idx, sentiment_result in zip(ids, sentiment_results):
            label = sentiment_result["label"]
            score = sentiment_result["score"]
            confidence_score = 1 - score if label == "NEGATIVE" else score

            mlflow.log_metric(
                f"News ID {idx} -  Positive Probability", confidence_score
            )
            mlflow.log_param(f"News ID {idx} - Predicted Label", label)

        mlflow.log_param("Model Name", MODEL)
        mlflow.log_param("Max Token Length", MAX_LENGTH)
        mlflow.log_param("Device", "GPU" if torch.cuda.is_available() else "CPU")
        mlflow.log_metric(
            "Average Inference Time", (end_time - start_time) / len(texts)
        )

    # push the sentiment results using xcom to store them in a Postgres table
    kwargs["ti"].xcom_push(key="sentiment_results", value=sentiment_results)
    kwargs["ti"].xcom_push(key="news_ids", value=ids)


def save_results_to_postgres(**kwargs):

    # pull the news texts and sentiment results
    sentiment_results = kwargs["ti"].xcom_pull(
        key="sentiment_results", task_ids="run_sentiment_pipeline_task"
    )
    news_items = kwargs["ti"].xcom_pull(key="news_items", task_ids="pull_news_task")
    ids = [item["news_id"] for item in news_items]
    texts = [item["text"] for item in news_items]

    # connect to postgres
    hook = PostgresHook(postgres_conn_id="news_records")
    conn = hook.get_conn()
    cursor = conn.cursor()

    # create a table if it doesn't exist
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

    # insert news texts and sentiment analysis results to the PostgreSQL table
    for idx, sentiment_result, news_text in zip(ids, sentiment_results, texts):
        predicted_label = sentiment_result["label"]

        cursor.execute(
            """
            INSERT INTO news_sentiment_results (news_id, topic, news_text, predicted_label)
            VALUES ( %s, %s, %s, %s);
            """,
            (idx, QUERY, news_text, predicted_label),
        )

    # commit the changes and close the cursor and connection
    conn.commit()
    cursor.close()
    conn.close()


# define the DAG
with DAG(
    dag_id="news_sentiment_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
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
