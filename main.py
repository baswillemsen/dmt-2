import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

from preprocess import load_train_val, load_test
from submission import make_submission

def evaluate_model(X_data, y_data, model):
    # Evaluate predictions
    gt_values = y_data.groupby('srch_id')['score'].apply(np.array).values
    predictions = (X_data.groupby('srch_id').apply(lambda x: model.predict(x.drop('srch_id', axis=1)))).values

    ndcg_score_list = []
    for gt_value, prediction in zip(gt_values, predictions):
        score = ndcg_score([gt_value], [prediction], k=5)
        ndcg_score_list.append(score)
    mean_score = np.mean(np.array(ndcg_score_list))
    print(mean_score)

def predict(model, df):
    return model.predict(df.drop('srch_id', axis=1))

def get_gt_values(df):
    return df['score']

def run(train_path, val_size):
    # Load the training, validation data
    X_train, y_train, X_val, y_val = load_train_val(train_path, val_size)
    groups = X_train.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

    # Train LambdaMART model
    model = XGBRanker(eval_metric='ndcg@5')
    model.fit(X_train.drop('srch_id', axis=1), y_train['score'], group=groups, verbose=True)

    # Evaluate training predictions
    print("NDCG@5 Score Train data: ")
    evaluate_model(X_train, y_train, model)
    # Evaluate validation predictions
    print("NDCG@5 Score Validation data: ")
    evaluate_model(X_val, y_val, model)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        help='Specifies location of the training data file')
    parser.add_argument('--val_size', type=float, default=0.33,
                        help='Specifies size of the validation set')

    args = parser.parse_args()
    model = run(args.train_path, args.val_size)

    make_pred = False
    if make_pred:
        X_test = load_test()
        make_submission(X_test, model)
    