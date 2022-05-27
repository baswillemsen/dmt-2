import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

from preprocessing import load_train

def train_model(X_train, y_train, model_name='pairwise', param_space={}):
    print("Training on columns: ", X_train.columns.values)
    groups = X_train.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    if model_name == 'pairwise':
        param_space['objective']='rank:pairwise'
    elif model_name == 'listwise_ndcg':
        param_space['objective']='rank:ndcg'
    elif model_name == 'listwise_map':
        param_space['objective']='rank:map'
    param_space['eval_metric']='ndcg@5'
    model = XGBRanker(**param_space)
    model.fit(X_train.drop('srch_id', axis=1), y_train['score'], group=groups, verbose=True)
    return model

def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])

def evaluate_model(X_data, y_data, model):
    # Evaluate predictions
    gt_values = y_data.groupby('srch_id')['score'].apply(np.array).values
    predictions = (X_data.groupby('srch_id').apply(lambda x: predict(model, x))).values

    ndcg_score_list = []
    for gt_value, prediction in zip(gt_values, predictions):
        score = ndcg_score([gt_value], [prediction], k=5)
        ndcg_score_list.append(score)
    mean_score = np.mean(np.array(ndcg_score_list))
    return mean_score

def run(train_path, model_name, param_space={}):
    # Load the training, validation data
    X_train, y_train, X_val, y_val = load_train(train_path, val_split=True)
    model = train_model(X_train, y_train, model_name, param_space)

    # Evaluate training predictions
    score = evaluate_model(X_train, y_train, model)
    print("NDCG@5 Score Train data:", score)
    # Evaluate validation predictions
    score = evaluate_model(X_val, y_val, model)
    print("NDCG@5 Score Validation data:", score)

    return model, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='pairwise', type=str,
                        help='What model to use for the hyperparameter tuning',
                        choices=['pairwise', 'listwise_ndcg', 'listwise_map'])
    parser.add_argument('--train_path', type=str, default='data/training_set_VU_DM.csv',
                    help='Specifies location of the training data file')
    parser.add_argument('--gpu_node', default=-1, type=int,
                        help='Specifies GPU node')

    args = parser.parse_args()
    #TODO set the param_space to the best hyperparameters resulting from hyperparameter search
    param_space = {'alpha': 2, 'colsample_by': 3, 'eta': 4, 'gamma': 0, 'lambda': 0, 'max_delta_step': 4, 'max_depth': 0, 'min_child_weight': 1, 'subsample': 1}
    if args.gpu_node != -1:
        param_space['gpu_id'] = args.gpu_node
        param_space['tree_method'] = 'gpu_hist'
    model = run(args.train_path, args.model_name, param_space)
    