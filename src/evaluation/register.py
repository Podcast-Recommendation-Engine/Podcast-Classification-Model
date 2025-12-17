




import mlflow
from mlflow.client import MlflowClient
from utils.logger import verbose_log


def register_best_model(test_results, tuned_run_ids, X_train):
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
    client = MlflowClient()
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