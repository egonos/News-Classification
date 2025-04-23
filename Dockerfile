FROM apache/airflow:2.8.1-python3.8

USER root
RUN apt-get update && apt-get install -y git

USER airflow
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121
