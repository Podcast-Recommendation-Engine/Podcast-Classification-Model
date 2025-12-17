import logging
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config.loader import load_data, load_models, set_tuning_param
from config.model_config import ANNOTATED_DATA_PATH, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from train.train import split_train_test, train_and_evaluate_model, tune_model
from utils import check_mlflow_connection, setup_common_logger, verbose_log

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
    baseline_models = load_models()
    
    verbose_log("Splitting data...")
    X_train, X_test, y_train, y_test = split_train_test(data)
    
    # 1. BASELINE TRAINING - Log CV metrics & models
    verbose_log("Training baseline models...")
    baseline_results = {}
    
    for model_name, model in baseline_models.items():
        verbose_log(f"Training: {model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("stage", "baseline_training")
            mlflow.set_tag("framework", "scikit-learn")

            results = train_and_evaluate_model(
                model, X_train, y_train, X_test, y_test, model_name
            )
            
            # Log validation/CV metrics (NOT test metrics)
            mlflow.log_metric("cv_accuracy", results['accuracy'])
            mlflow.log_metric("cv_precision", results['precision'])
            mlflow.log_metric("cv_recall", results['recall'])
            mlflow.log_metric("cv_f1_score", results['f1_score'])
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            baseline_results[model_name] = results

    # 2. HYPERPARAMETER TUNING - Log best params & tuned models
    verbose_log("Tuning models...")
    tuned_models = {}
    tuned_run_ids = {}  # Track run IDs for model registration
    
    param_grids = set_tuning_param()
    for model_name in ['Logistic Regression', 'Linear SVM']:
        
        with mlflow.start_run(run_name=f"{model_name}_tuned") as run:
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("stage", "hyperparameter_tuning")
            
            best_model, best_params, best_score, n_combinations = tune_model(
                baseline_models[model_name], param_grids[model_name], X_train, y_train
            )
            
            # Log best hyperparameters
            for param, value in best_params.items():
                mlflow.log_param(param, value)
            
            # Log CV score
            mlflow.log_metric("best_cv_f1_score", best_score)
            mlflow.log_metric("n_combinations_tested", n_combinations)
            
            # Log tuned model
            mlflow.sklearn.log_model(best_model, "model")
            
            tuned_models[model_name] = best_model
            tuned_run_ids[model_name] = run.info.run_id  # Save run ID

    # 3. FINAL TEST EVALUATION - Single run comparing all models
    verbose_log("Evaluating on test set...")
    test_results = {}
    
    with mlflow.start_run(run_name="final_test_evaluation"):
        mlflow.set_tag("stage", "test_evaluation")
        
        # Log dataset sizes
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Evaluate all tuned models
        for model_name, model in tuned_models.items():
            y_pred = model.predict(X_test)
            
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            test_results[model_name] = test_f1
            
            # Log test metrics with model prefix
            mlflow.log_metric(f"test_{model_name}_accuracy", 
                            accuracy_score(y_test, y_pred))
            mlflow.log_metric(f"test_{model_name}_precision", 
                            precision_score(y_test, y_pred, zero_division=0))
            mlflow.log_metric(f"test_{model_name}_recall", 
                            recall_score(y_test, y_pred, zero_division=0))
            mlflow.log_metric(f"test_{model_name}_f1_score", test_f1)
    
    # 4. REGISTER BEST MODEL
    verbose_log("Registering best model...")
    
    # Find best model based on test F1 score
    best_model_name = max(test_results, key=test_results.get)
    best_f1_score = test_results[best_model_name]
    best_run_id = tuned_run_ids[best_model_name]
    
    verbose_log(f"Best model: {best_model_name} (F1: {best_f1_score:.4f})")
    
    # Register the model
    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = "podcast-kid-friendly-classifier"
    
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name
    )
    
    verbose_log(f"Model registered: {registered_model_name} v{model_version.version}")
    
    # Transition to Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True  # Archive previous production models
    )
    
    verbose_log(f"Model v{model_version.version} transitioned to Production")
    
    # Add model description
    client.update_model_version(
        name=registered_model_name,
        version=model_version.version,
        description=f"Best model: {best_model_name} | Test F1: {best_f1_score:.4f} | Trained on {len(X_train)} samples"
    )
    
    verbose_log("Training completed!")

if __name__ == "__main__":
    setup_common_logger()
    main()

