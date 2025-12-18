import logging
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold

from utils.logger import verbose_log



def tune_model(model, param_grid, X_train, Y_train, tuning_scoring, random_state, cv=5):
    n_combinations = len(list(ParameterGrid(param_grid)))
    grid_search= GridSearchCV(
        model,
        param_grid,
        cv= StratifiedKFold(cv, shuffle=True, random_state= random_state),
        scoring=tuning_scoring,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train,  Y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, n_combinations


def set_tuning_param():
    # Grille de parametres
    param_grids = {
    'Logistic Regression': {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 2, 3],
        'tfidf__max_df': [0.9, 0.95, 1.0],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__l1_ratio': [0],  
        'clf__class_weight': ['balanced']
    },
    
    'Linear SVM': {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 2, 3],
        'tfidf__max_df': [0.9, 0.95, 1.0],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__class_weight': ['balanced']
    }
    }
    return param_grids


def run_tuning_model(baseline_models, X_train, Y_train, tuning_scoring, random_state, cv):
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
                baseline_models[model_name], param_grids[model_name], X_train, Y_train, tuning_scoring,
                random_state, cv
            )
            
            # Log best hyperparameters
            for param, value in best_params.items():
                mlflow.log_param(param, value)
            
            # Log CV score
            mlflow.log_metric("best_cv_f1_score", best_score)
            mlflow.log_metric("n_combinations_tested", n_combinations)
            
            # Log tuned model with input example
            import pandas as pd
            input_example = pd.DataFrame(X_train[:5]) if len(X_train) > 5 else pd.DataFrame(X_train)
            mlflow.sklearn.log_model(best_model, artifact_path="model", input_example=input_example)
            
            tuned_models[model_name] = best_model
            tuned_run_ids[model_name] = run.info.run_id  # Save run ID
    return tuned_models, tuned_run_ids

