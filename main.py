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

def run():
    # Load the data
    X_train, y_train, X_val, y_val = load_train_val()
    # Train LambdaMART model
    model = XGBRanker(eval_metric='ndcg@5')
    model.fit(X_train, y_train, qid=X_train['srch_id'])

    # # Evaluate predictions
    ndcg_scores = []
    for srch_id, group in X_val.groupby(['srch_id']):
        scores = model.predict(group)
        indices = X_val[X_val['srch_id'] == srch_id].index.tolist()

        score = ndcg_score([scores], [y_val[indices].values], k=5)
        ndcg_scores.append(score)
    print("NCDG@5 Validation Score:", np.array(ndcg_scores).mean())

    # make_submission(model)

if __name__ == "__main__":
    run()