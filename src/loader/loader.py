import logging
from mlflow.sklearn import log_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils.logger import setup_common_logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier


def load_models(mindf, maxdf):
    logging.info("Configuring TF-IDF parameters...")
    tfidf_params = {
        'max_features': None,
        'min_df': mindf,
        'max_df': maxdf,
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
                l1_ratio=0 
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



