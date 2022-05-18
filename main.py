import numpy as np
import pandas as pd

from preprocess import train_val_split, load_test

from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

def make_submission(model):
    X_test = load_test()

def run():
    # Load the data
    X_train, y_train, X_val, y_val = train_val_split()
    # Train LambdaMART model
    model = XGBRanker(rank='ndcg', eval_metric='ndcg@5')
    model.fit(X_train)

    # Evaluate predictions
    predictions = model.predict(X_train)
    score = ndcg_score(predictions, y_train)
    print("NCDG@5 Train Score:", score)

    predictions = model.predict(X_val)
    score = ndcg_score(predictions, y_val)
    print("NCDG@5 Validation Score:", score)

if __name__ == "__main__":
    run()