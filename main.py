import numpy as np
import pandas as pd

from preprocess import load_train_val, load_test

from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

def make_submission(model):
    X_test = load_test()
    with open('submission.csv', 'w') as fout:
        fout.write("srch_id,prop_id\n")
        for srch_id, group in X_test.groupby(['srch_id']):
            prop_ids = make_ranking(group, model)
            for prop_id in prop_ids:
                fout.write(f"{srch_id},{prop_id}\n")
    print("Made submission")

def make_ranking(X_test, model):
    predictions = model.predict(X_test)
    sorted_indices = np.argsort(predictions)
    prop_ids = X_test['prop_id'].iloc[sorted_indices]
    return prop_ids

def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])

def get_gt_values(df):
    return df['score']

def run():
    # Load the data
    X_train, y_train, X_val, y_val = load_train_val()
    groups = X_train.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    x_train_values = X_train.drop(['srch_id'], axis=1)
    # Train LambdaMART model
    model = XGBRanker(eval_metric='ndcg@5')
    y_train_scores = y_train['score']
    model.fit(x_train_values, y_train_scores, group=groups, verbose=True)

    # Evaluate predictions
    gt_values = y_train.groupby('srch_id')['score'].apply(np.array).values
    predictions = (X_train.groupby('srch_id').apply(lambda x: predict(model, x))).values
    # prop_id_predictions = X_train.groupby('srch_id')['prop_id'].apply(np.array).values

    ndcg_score_list = []
    for i in range(len(predictions)):
        score = ndcg_score(gt_values[i].reshape(1,-1), predictions[i].reshape(1,-1), k=5)
        ndcg_score_list.append(score)
    mean_score = np.mean(np.array(ndcg_score_list))
    print("NDCG@5 Train Score:", mean_score)

    gt_values = y_val.groupby('srch_id')['score'].apply(np.array).values
    predictions = (X_val.groupby('srch_id').apply(lambda x: predict(model, x))).values
    ndcg_score_list = []
    for i in range(len(predictions)):
        score = ndcg_score(gt_values[i].reshape(1,-1), predictions[i].reshape(1,-1), k=5)
        ndcg_score_list.append(score)
    mean_score = np.mean(np.array(ndcg_score_list))
    print("NDCG@5 Validation Score:", mean_score)

    return model

if __name__ == "__main__":
    make_pred = True

    model = run()
    if make_pred:
        make_submission(model)
    