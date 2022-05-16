import numpy as np
import pandas as pd

from preprocess import train_test_split

from sklearn.metrics import ndcg_score
from xgboost import XGBRanker


def run():
    # LambdaMART model
    X_train, y_train, X_test, y_test = train_test_split
    model = XGBRanker(rank='ndcg')
    model.fit(X_train)
    predictions = model.predict(X_test)
    score = ndcg_score(predictions, y_true)
    print("NCDG@5")