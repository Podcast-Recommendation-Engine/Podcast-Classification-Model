
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.logger import verbose_log




def evaluate_test_set(evaluation_test_set, tuned_models, X_train, X_test, y_test):
    verbose_log("Evaluating on test set...")
    test_results = {}
    
    with mlflow.start_run(run_name=evaluation_test_set):
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
                            float(accuracy_score(y_test, y_pred)))
            
            mlflow.log_metric(f"test_{model_name}_precision", 
                            float(precision_score(y_test, y_pred, zero_division=0)))
            
            mlflow.log_metric(f"test_{model_name}_recall", 
                            float(recall_score(y_test, y_pred, zero_division=0)))
            
            mlflow.log_metric(f"test_{model_name}_f1_score", float(test_f1))
    return test_results