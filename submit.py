import numpy as np

from evaluation import predict

def make_ranking(X_test, model):
    predictions = predict(model, X_test)
    sorted_indices = np.argsort(predictions)
    prop_ids = X_test['prop_id'].iloc[sorted_indices]
    return prop_ids

def make_submission(X_test, model):
    with open('data/submission.csv', 'w') as fout:
        fout.write("srch_id,prop_id\n")
        for srch_id, group in X_test.groupby(['srch_id']):
            prop_ids = make_ranking(group, model)
            for prop_id in prop_ids:
                fout.write(f"{srch_id},{prop_id}\n")
    print("Made submission")

