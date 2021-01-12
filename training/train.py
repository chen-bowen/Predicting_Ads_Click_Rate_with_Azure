# Modified from https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/

import argparse
import os

# importing necessary libraries
import numpy as np
from utils import reduce_mem, unique_count

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgbm

import joblib

from azureml.core.run import Run
from azureml.core import Dataset

import time


run = Run.get_context()
ws = run.experiment.workspace
key = "CTR-Training"


def preprocess_data(ds):
    """ Perform brief data preprocessing for the incoming dataset object """

    # convert the dataset object to dataframe
    df = ds.to_pandas_dataframe()

    # categorical feature unique count
    categorical_cols = [
        "slot_id",
        "adv_id",
        "adv_prim_id",
        "creat_type_cd",
        "inter_type_cd",
        "age",
        "city",
        "uid",
        "dev_id",
        "task_id",
    ]
    df = unique_count(df, categorical_cols)
    df = reduce_mem(df)

    # drop engineered features
    drop_fea = ["pt_d", "communication_onlinerate", "uid"]
    df.drop(columns=drop_fea, inplace=True)

    x_df = df.drop(columns=["label"])
    y_df = df.pop("label")

    return x_df, y_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="learning rate for lightGBM model",
    )

    parser.add_argument(
        "--num_leaves",
        type=int,
        default=31,
        help="max number of leaves in one tree, lower means less overfitting",
    )

    parser.add_argument(
        "--boosting",
        type=str,
        default="gbdt",
        help="lightGBM boosting type, choices are gbdt, rf, dart, goss",
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=-1,
        help="limit the max depth for tree model, This is used to deal with over-fitting when #data is small",
    )

    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=0.0,
        help="L1 regularization, higher means lower overfitting",
    )

    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=0.0,
        help="L2 regularization, higher means lower overfitting",
    )

    parser.add_argument(
        "--path_smooth",
        type=float,
        default=0,
        help="controls smoothing applied to tree nodes",
    )

    parser.add_argument(
        "--max_bin",
        type=int,
        default=255,
        help="max number of bins that feature values will be bucketed in",
    )

    args = parser.parse_args()
    run.log("Learning rate", np.float(args.learning_rate))
    run.log("Number of leaves", np.float(args.num_leaves))
    run.log("Boosting method", np.str(args.boosting))
    run.log("Max depth of trees", np.int(args.max_depth))
    run.log("lambda l1", np.float(args.lambda_l1))
    run.log("lambda l2", np.float(args.lambda_l2))
    run.log("path smoothing", np.float(args.path_smooth))
    run.log("max number of bins", np.float(args.max_bin))

    # loading the building structure dataset from registered dataset
    ctr_ds = Dataset.get_by_name(ws, name=key)

    # X -> features, y -> label
    X, y = preprocess_data(ctr_ds)

    # dividing X, y into train and test data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # training lightGBM classifier

    light_gbm = lgbm.LGBMClassifier(
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        boosting=args.boosting,
        max_depth=args.max_depth,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        path_smooth=args.path_smooth,
        max_bin=args.max_bin,
    ).fit(X_train, y_train)

    lgbm_predictions = light_gbm.predict(X_val)

    # model accuracy for X_val
    accuracy = light_gbm.score(X_val, y_val)
    print("Accuracy of LGBM classifier on test set: {:.2f}".format(accuracy))
    run.log("Accuracy", np.float(accuracy))

    # model auc for X_val
    pred_prob = light_gbm.predict_proba(X_val)
    auc_score = roc_auc_score(y_val, pred_prob[:, 1], average="weighted")
    run.log("AUC_weighted", np.float(auc_score))
    print("AUC score of LGBM model on test set: {:.2f}".format(auc_score))

    # creating a confusion matrix
    cm = confusion_matrix(y_val, lgbm_predictions)
    print(cm)

    os.makedirs("outputs", exist_ok=True)

    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(light_gbm, "outputs/model.joblib")


if __name__ == "__main__":
    main()
