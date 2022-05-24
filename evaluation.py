import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

from preprocessing import load_train

def train_model(X_train, y_train):
    groups = X_train.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    model = XGBRanker(eval_metric='ndcg@5')
    model.fit(X_train.drop('srch_id', axis=1), y_train['score'], group=groups, verbose=True)
    return model

def evaluate_model(X_data, y_data, model):
    # Evaluate predictions
    gt_values = y_data.groupby('srch_id')['score'].apply(np.array).values
    predictions = (X_data.groupby('srch_id').apply(lambda x: model.predict(x.drop('srch_id', axis=1)))).values

    ndcg_score_list = []
    for gt_value, prediction in zip(gt_values, predictions):
        score = ndcg_score([gt_value], [prediction], k=5)
        ndcg_score_list.append(score)
    mean_score = np.mean(np.array(ndcg_score_list))
    return mean_score

def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])

def run(train_path):
    # Load the training, validation data
    X_train, y_train, X_val, y_val = load_train(train_path, val_split=True)
    model = train_model(X_train, y_train)

    # Evaluate training predictions
    score = evaluate_model(X_train, y_train, model)
    print("NDCG@5 Score Train data:", score)
    # Evaluate validation predictions
    score = evaluate_model(X_val, y_val, model)
    print("NDCG@5 Score Validation data:", score)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        help='Specifies location of the training data file')
                        
    args = parser.parse_args()
    model = run(args.train_path)
    