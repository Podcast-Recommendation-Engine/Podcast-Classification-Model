import os
from dotenv import load_dotenv

load_dotenv()


ANNOTATED_DATA_PATH= "data/annotated/podcasts_annotated.csv"
MLFLOW_URL= os.getenv("MLFLOW_URL", "mlflow")
MLFLOW_PORT= int(os.getenv("MLFLOW_PORT", 5000))
MLFLOW_TRACKING_URI= f"http://{MLFLOW_URL}:{MLFLOW_PORT}"
EXPERIMENT_NAME= os.getenv("EXPERIMENT_NAME", "podcast-classification-kid-friendly")

EVALUATING_TEST_SET= os.getenv("EVALUATING_TEST_SET", "podcast-classification-kid-friendly")