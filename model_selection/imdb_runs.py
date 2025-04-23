from utils import *
import subprocess

#dataset preparation
subprocess.run(["wget", "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"],
               stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
subprocess.run(["tar", "-xvzf", "aclImdb_v1.tar.gz"],
               stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

TEST_DF = load_imdb_data("aclImdb", "test")
TEST_DF = TEST_DF.sample(frac=0.5,random_state=42,replace = False).reset_index(drop=True)

#model runs
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTHS = [512,256,128]
LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}

for MAX_LENGTH in MAX_LENGTHS:
  run_model(MODEL_NAME=MODEL_NAME, MAX_LENGTH = MAX_LENGTH, LABEL_MAP=LABEL_MAP, TEST_DF=TEST_DF, DF_NAME="IMDB")


MODEL_NAME = "AirrStorm/DistilBERT-SST2-Yelp"
LABEL_MAP = {"LABEL_1": 1, "LABEL_0": 0}

for MAX_LENGTH in MAX_LENGTHS:
  run_model(MODEL_NAME=MODEL_NAME, MAX_LENGTH = MAX_LENGTH, LABEL_MAP=LABEL_MAP, TEST_DF=TEST_DF,DF_NAME="IMDB")

MODEL_NAME = "siebert/sentiment-roberta-large-english"
LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}

for MAX_LENGTH in MAX_LENGTHS:
  run_model(MODEL_NAME=MODEL_NAME, MAX_LENGTH = MAX_LENGTH, LABEL_MAP=LABEL_MAP, TEST_DF=TEST_DF,DF_NAME="IMDB")