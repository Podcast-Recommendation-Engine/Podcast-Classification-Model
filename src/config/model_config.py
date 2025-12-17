from dotenv import load_dotenv

load_dotenv()

 
ANNOTATED_DATA_PATH= "data/annotated/podcasts_annotated.csv"
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Changed from localhost to mlflow
EXPERIMENT_NAME= "podcast-classification-kid-friendly"