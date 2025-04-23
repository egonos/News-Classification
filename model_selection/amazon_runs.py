from utils import load_amazon_data, run_model


TEST_DF = load_amazon_data()

MAX_LENGTHS = [512, 256, 128]
DF_NAME = "Amazon Reviews"


LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
for MAX_LENGTH in MAX_LENGTHS:
    run_model(
        MODEL_NAME=MODEL_NAME,
        MAX_LENGTH=MAX_LENGTH,
        LABEL_MAP=LABEL_MAP,
        TEST_DF=TEST_DF,
        DF_NAME="AMAZON",
    )

MODEL_NAME = "AirrStorm/DistilBERT-SST2-Yelp"
LABEL_MAP = {"LABEL_1": 1, "LABEL_0": 0}
for MAX_LENGTH in MAX_LENGTHS:
    run_model(
        MODEL_NAME=MODEL_NAME,
        MAX_LENGTH=MAX_LENGTH,
        LABEL_MAP=LABEL_MAP,
        TEST_DF=TEST_DF,
        DF_NAME="AMAZON",
    )


MODEL_NAME = "siebert/sentiment-roberta-large-english"
LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}
for MAX_LENGTH in MAX_LENGTHS:
    run_model(
        MODEL_NAME=MODEL_NAME,
        MAX_LENGTH=MAX_LENGTH,
        LABEL_MAP=LABEL_MAP,
        TEST_DF=TEST_DF,
        DF_NAME="AMAZON",
    )
