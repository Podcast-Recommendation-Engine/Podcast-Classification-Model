




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
    
    # Check if model artifacts are already registered
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = "podcast-kid-friendly-classifier"
    
    # Check if this run is already registered
    model_already_registered = False
    try:
        registered_model = client.get_registered_model(registered_model_name)
        verbose_log(f"Model '{registered_model_name}' exists in registry")
        
        # Check if this specific run_id is already registered as a version
        for version in client.search_model_versions(f"name='{registered_model_name}'"):
            if version.run_id == best_run_id:
                verbose_log(f"Model artifacts from run {best_run_id} already registered as version {version.version}")
                model_version = version
                model_already_registered = True
                break
    except Exception as e:
        verbose_log(f"Model '{registered_model_name}' not found in registry, creating new model")
    
    # Register the model if not already registered
    if not model_already_registered:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )
        verbose_log(f"Model registered: {registered_model_name} v{model_version.version}")
    else:
        verbose_log(f"Using existing model version: {model_version.version}")
    
    # Transition to Production
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