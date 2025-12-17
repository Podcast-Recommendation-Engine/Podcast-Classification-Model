
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from config.loader import load_data, load_models, set_tuning_param
from config.model_config import ANNOTATED_DATA_PATH, EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, 
                             model_name):
    
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


def split_train_test(data: pd.DataFrame, test_size=0.2, random_state=42):
    logging.info("Extracting features and labels...")
    X = data['keywords_text']
    y = data['is_kid_friendly']
    logging.info(f"Splitting data: test_size={test_size}, stratified=True")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logging.info(f"Split complete: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test



def tune_model(model, param_grid, X_train, Y_train, cv=5):
    n_combinations = len(list(ParameterGrid(param_grid)))
    grid_search= GridSearchCV(
        model,
        param_grid,
        cv= StratifiedKFold(cv, shuffle=True, random_state= 42),
        scoring='f1',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train,  Y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, n_combinations


