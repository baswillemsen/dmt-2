import numpy as np
import pandas as pd

from preprocess import load_train_val, load_test
from submission import make_submission, make_ranking

from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

def evaluate_model(X_data, y_data, model):
    # Evaluate predictions
    gt_values = y_data.groupby('srch_id')['score'].apply(np.array).values
    predictions = (X_data.groupby('srch_id').apply(lambda x: predict(model, x))).values
    # prop_id_predictions = X_train.groupby('srch_id')['prop_id'].apply(np.array).values

    ndcg_score_list = []
    for i in range(len(predictions)):
        score = ndcg_score(gt_values[i].reshape(1, -1), predictions[i].reshape(1, -1), k=5)
        ndcg_score_list.append(score)
    mean_score = np.mean(np.array(ndcg_score_list))
    print(mean_score)

def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])

def get_gt_values(df):
    return df['score']

def run():
    # Load the training, validation data
    X_train, y_train, X_val, y_val = load_train_val()
    groups = X_train.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    x_train_values = X_train.drop(['srch_id'], axis=1)
    # Train LambdaMART model
    model = XGBRanker(eval_metric='ndcg@5')
    y_train_scores = y_train['score']
    model.fit(x_train_values, y_train_scores, group=groups, verbose=True)

    # Evaluate training predictions
    print("NDCG@5 Score Train data: ")
    evaluate_model(X_train, y_train, model)
    # Evaluate validation predictions
    print("NDCG@5 Score Validation data: ")
    evaluate_model(X_val, y_val, model)

    return model

if __name__ == "__main__":
    make_pred = True

    model = run()
    if make_pred:
        X_test = load_test()
        make_submission(X_test, model)
    