import logging
import time
import mlflow
from config.mlflow import ANNOTATED_DATA_PATH, EVALUATING_TEST_SET, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from config.model import RANDOM_STATE, TFIDF_MAX_DF, TFIDF_MIN_DF, TUNING_CROSS_VALIDATION_FOLDS, TUNING_SCORING
from evaluation.evaluate import evaluate_test_set
from evaluation.register import register_best_model
from loader.loader import load_data, load_models
from training.baseline import run_train_baseline_model
from training.tuning import run_tuning_model
from utils.logger import setup_common_logger, verbose_log
from utils.mlflow import check_mlflow_connection, split_data


def main():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    if not check_mlflow_connection(MLFLOW_TRACKING_URI):
        logging.error("MLflow not accessible")
        return
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    verbose_log(f"MLflow connected: {MLFLOW_TRACKING_URI}")
    
    verbose_log("Loading data...")
    data = load_data(ANNOTATED_DATA_PATH)
    verbose_log(f"Loaded {len(data)} examples")
    
    verbose_log("Loading models...")
    baseline_models = load_models(TFIDF_MIN_DF, TFIDF_MAX_DF)
    
    verbose_log("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data)
    
    baseline_results= run_train_baseline_model(baseline_models, X_train, X_test, y_train, y_test)

    tuned_models, tuned_run_ids= run_tuning_model(baseline_models, X_train, y_train, TUNING_SCORING, RANDOM_STATE, TUNING_CROSS_VALIDATION_FOLDS)
    test_results= evaluate_test_set(EVALUATING_TEST_SET, tuned_models, X_train, X_test, y_test)
    
    register_best_model(test_results, tuned_run_ids, X_train)

    verbose_log("Training completed!")

if __name__ == "__main__":
    setup_common_logger()
    main()

