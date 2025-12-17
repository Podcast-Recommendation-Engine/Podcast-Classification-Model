import logging
from mlflow.sklearn import log_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from config.model_config import ANNOTATED_DATA_PATH, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from utils import setup_common_logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier


def load_models():
    logging.info("Configuring TF-IDF parameters...")
    tfidf_params = {
        'max_features': None,
        'min_df': 2,
        'max_df': 0.95,
        'ngram_range': (1, 2),
        'sublinear_tf': True
    }

    logging.info("Creating model pipelines...")
    baseline_models = {
        'Dummy (Baseline Naïve)': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', DummyClassifier(strategy='most_frequent', random_state=42))
        ]),
        
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', LogisticRegression(
                max_iter=1000, random_state=42, 
                class_weight='balanced', solver='liblinear',
                l1_ratio=0  # Replaces penalty='l2'
            ))
        ]),
        
        'Linear SVM': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', LinearSVC(
                max_iter=2000, random_state=42,
                class_weight='balanced', dual='auto'
            ))
        ]),
        
        'Multinomial Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', MultinomialNB(alpha=1.0))
        ])
    }
    logging.info(f"Loaded {len(baseline_models)} models")
    return baseline_models



def load_data(path) -> pd.DataFrame:
    logging.info(f"Reading CSV from: {path}")
    df = pd.read_csv(path)
    logging.info(f"Creating keywords_text column...")
    df['keywords_text'] = df['keywords_clean'].apply(
        lambda x: ' '.join(eval(x)) if isinstance(x, str) else ' '.join(x)
    )
    logging.info(f"Data prepared: {len(df)} rows, {len(df.columns)} columns")
    return df



def set_tuning_param():
    # Grille de parametres
    param_grids = {
    'Logistic Regression': {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 2, 3],
        'tfidf__max_df': [0.9, 0.95, 1.0],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__l1_ratio': [0],  # Replaces penalty=['l2']
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

