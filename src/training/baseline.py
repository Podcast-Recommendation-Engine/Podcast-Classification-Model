
import logging
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.logger import verbose_log



def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    
    logging.info("Training model...")
    model.fit(X_train, y_train)

    logging.info("Making predictions...")
    y_pred = model.predict(X_test)

    logging.info("Calculating metrics...")
    results = {
        'model': model,
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'y_pred': y_pred
    }

    logging.info(f"Accuracy: {results['accuracy']:.4f}, F1-Score: {results['f1_score']:.4f}")
    return results




def run_train_baseline_model(baseline_models, X_train, X_test, y_train, y_test):
# 1. BASELINE TRAINING - Log CV metrics & models
    verbose_log("Training baseline models...")
    baseline_results = {}
    
    for model_name, model in baseline_models.items():
        verbose_log(f"Training: {model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("stage", "baseline_training")
            mlflow.set_tag("framework", "scikit-learn")

            results = train_and_evaluate(
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
    return baseline_results


