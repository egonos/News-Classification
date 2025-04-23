from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import os
import time
from tqdm import tqdm
import kagglehub
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def load_imdb_data(data_dir, subset):
    data = []
    labels = {"pos": 1, "neg": 0}

    for label in ["pos", "neg"]:
        path = os.path.join(data_dir, subset, label)
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                with open(os.path.join(path, filename), encoding="utf-8") as file:
                    review = file.read()
                    data.append((review, labels[label]))

    df = pd.DataFrame(data, columns=["review", "label"])
    return df


def load_amazon_data():
    path = kagglehub.dataset_download("kritanjalijain/amazon-reviews")
    csv_path = os.path.join(path, "train.csv")
    df = pd.read_csv(csv_path, header=None)
    df = df.drop(1, axis=1)
    df = df[[2, 0]]
    df.rename(columns={2: "text", 0: "label"}, inplace=True)
    df["label"] = df["label"] - 1
    negative_samples = df[df["label"] == 0].sample(5000, random_state=42)
    positive_samples = df[df["label"] == 1].sample(3000, random_state=42)
    df = (
        pd.concat([negative_samples, positive_samples], ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )
    return df


def load_reviews_data():
    data_path = (
        "/root/.cache/kagglehub/datasets/ahmedabdulhamid/reviews-dataset/versions/1"
    )
    positive_file_path = os.path.join(data_path, "TrainingDataPositive.txt")
    negative_file_path = os.path.join(data_path, "TrainingDataNegative.txt")

    with open(positive_file_path, "r", encoding="utf-8") as f:
        positive = f.readlines()

    with open(negative_file_path, "r", encoding="utf-8") as f:
        negative = f.readlines()

    df_pos = pd.DataFrame({"text": [line.strip() for line in positive], "label": 1})
    df_neg = pd.DataFrame({"text": [line.strip() for line in negative], "label": 0})
    df = (
        pd.concat([df_pos, df_neg], ignore_index=True)
        .sample(frac=1, random_state=42, replace=False)
        .reset_index(drop=True)
    )
    return df


def run_model(MODEL_NAME, MAX_LENGTH, LABEL_MAP, TEST_DF, DF_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=MAX_LENGTH,
        truncation=True,
    )

    texts = TEST_DF.iloc[:, 0].values
    true_labels = TEST_DF.iloc[:, 1].values

    logs = []
    batch_size = 32

    start_time = time.time()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        output = sentiment_pipeline(batch.tolist(), truncation=True)
        logs.extend(output)
    end_time = time.time()

    labels = list(map(LABEL_MAP.get, [log["label"] for log in logs]))
    positive_probs = [
        (
            log["score"]
            if (log["label"] == "POSITIVE") or (log["label"] == "LABEL_1")
            else 1 - log["score"]
        )
        for log in logs
    ]
    acc = accuracy_score(y_true=true_labels, y_pred=labels)
    f1 = f1_score(y_true=true_labels, y_pred=labels)
    binary_log_loss = log_loss(y_true=true_labels, y_pred=positive_probs)
    roc_auc = roc_auc_score(y_true=true_labels, y_score=positive_probs)
    average_inference_time = (end_time - start_time) / len(texts)

    print("Model Name: ", MODEL_NAME)
    print("Token Max Length: ", MAX_LENGTH)
    print("Dataset: ", "IMDB")
    print("Pos-Neg Ratio: ", TEST_DF.iloc[:, 1].value_counts(normalize=True)[1])
    print("Accuracy: ", acc)
    print("F1 Score: ", f1)
    print("Binary Log Loss: ", binary_log_loss)
    print("ROC-AUC Score: ", roc_auc)
    print("Average Inference Time: ", average_inference_time)
    print("Device: ", "GPU" if torch.cuda.is_available() else "CPU")
    print()

    model_logs = pd.DataFrame(
        {
            "Model Name": [MODEL_NAME],
            "Token Max Length": [MAX_LENGTH],
            "Dataset": DF_NAME,
            "Pos-Neg Ratio": TEST_DF.iloc[:, 1].value_counts(normalize=True)[1],
            "Accuracy": [acc],
            "F1 Score": [f1],
            "Binary Log Loss": [binary_log_loss],
            "ROC-AUC Score": [roc_auc],
            "Inference Time": [average_inference_time],
        }
    ).round(4)

    base_dir = os.path.dirname(__file__)
    logs_path = os.path.join(base_dir, "model_logs.csv")

    if os.path.exists(logs_path):
        existing_logs = pd.read_csv(logs_path)
        updated_logs = pd.concat([existing_logs, model_logs], ignore_index=True)
    else:
        updated_logs = model_logs

    updated_logs.to_csv(logs_path, index=False)
