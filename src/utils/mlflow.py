
import logging
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split


def check_mlflow_connection(uri):
    try:
        mlflow.set_tracking_uri(uri)
        client = mlflow.MlflowClient()
        client.search_experiments()
        return True
    except Exception as e:
        logging.error(f"MLflow connection failed: {e}")
        return False



def split_data(data: pd.DataFrame, test_size=0.2, random_state=42):
    logging.info("Extracting features and labels...")
    X = data['keywords_text']
    y = data['is_kid_friendly']
    logging.info(f"Splitting data: test_size={test_size}, stratified=True")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logging.info(f"Split complete: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test
