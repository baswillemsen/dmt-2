import numpy as np
import pandas as pd

from preprocess import load_train_val, load_test

from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

def make_submission(model):
    X_test = load_test()

def run():
    # Load the data
    X_train, y_train, X_val, y_val = load_train_val()
    # Train LambdaMART model
    model = XGBRanker(eval_metric='ndcg@5')
    model.fit(X_train, y_train, qid=X_train['srch_id'])

    # Evaluate predictions
    predictions = model.predict(X_train)
    print(predictions)
    score = ndcg_score(predictions, y_train)
    print("NCDG@5 Train Score:", score)

    predictions = model.predict(X_val)
    score = ndcg_score(predictions, y_val)
    print("NCDG@5 Validation Score:", score)

if __name__ == "__main__":
    run()