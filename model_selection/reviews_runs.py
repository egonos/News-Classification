from utils import load_reviews_data, run_model

TEST_DF = load_reviews_data()

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTHS = [512, 256, 128]
LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}

for MAX_LENGTH in MAX_LENGTHS:
    run_model(
        MODEL_NAME=MODEL_NAME,
        MAX_LENGTH=MAX_LENGTH,
        LABEL_MAP=LABEL_MAP,
        TEST_DF=TEST_DF,
        DF_NAME="REVIEWS",
    )


MODEL_NAME = "AirrStorm/DistilBERT-SST2-Yelp"
LABEL_MAP = {"LABEL_1": 1, "LABEL_0": 0}

for MAX_LENGTH in MAX_LENGTHS:
    run_model(
        MODEL_NAME=MODEL_NAME,
        MAX_LENGTH=MAX_LENGTH,
        LABEL_MAP=LABEL_MAP,
        TEST_DF=TEST_DF,
        DF_NAME="REVIEWS",
    )

MODEL_NAME = "siebert/sentiment-roberta-large-english"
LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}

for MAX_LENGTH in MAX_LENGTHS:
    run_model(
        MODEL_NAME=MODEL_NAME,
        MAX_LENGTH=MAX_LENGTH,
        LABEL_MAP=LABEL_MAP,
        TEST_DF=TEST_DF,
        DF_NAME="REVIEWS",
    )
