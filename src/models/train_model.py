from src.metrics import metrics_report, plot_confusion_matrix, acc_per_class
from src.preprocessing import load_split_data_as_df, construct_train_data
from src.utils import print_info, print_warning

import argparse
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from statistics import mean, stdev
from tqdm import tqdm
from xgboost import XGBClassifier


def pipeline(x, y,
             x_test, y_test, split_by, encoding,
             df_result, models,
             save_dir: str, dataset: str,
             n_splits=10, n_repeats=5, debug: bool = False):
    # Create StratifiedKFold object.
    skf = RepeatedStratifiedKFold(n_splits=n_splits,
                                  n_repeats=n_repeats,
                                  random_state=42)

    # Apply the pipeline for each model
    for clf_name, clf in models.items():
        print_warning(f"Dataset: {dataset}")
        print_info(f"\n[INFO]Starting Cross-Validation with: {skf}"
                   f"\nIn total {n_splits * n_repeats} models(`n_repeats`*`n_splits`) will be evaluated which are of the following model class:")
        print_warning(clf)

        scores = defaultdict(list)
        scores_test = defaultdict(list)
        for train_index, val_index in tqdm(skf.split(x, y),
                                           total=n_splits * n_repeats):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]
            # Fit data to the model
            clf.fit(x_train, y_train)
            # Calculate ROC-AUC score
            scores["ROC"].append(roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1]))
            # Calculate accuracy
            scores["Acc"].append(clf.score(x_val, y_val))

        # Print the output.
        print(f'\nMax Acc : {max(scores["Acc"]) * 100 :.2f}',   f'\tMax ROC : {max(scores["ROC"]) * 100 :.2f}',
              f'\nMin Acc : {min(scores["Acc"]) * 100 :.2f}',   f'\tMin ROC : {min(scores["ROC"]) * 100 :.2f}',
              f'\nMean Acc: {mean(scores["Acc"]) * 100 :.2f}',  f'\tMean ROC: {mean(scores["ROC"]) * 100 :.2f}',
              f'\nStDev   : {stdev(scores["Acc"]) * 100 :.3f}', f'\tStDev   : {stdev(scores["ROC"]) * 100 :.3f}')

        print_info("\n[INFO]Training the model with all training data...")
        clf.fit(x, y)
        print_info(f"\n[INFO]Saving the model to: {save_dir}...")
        model_name = str(clf).split('(')[0]
        _ = joblib.dump(clf, f"{save_dir}/{dataset}_{split_by}_{model_name}_{encoding}.joblib")

        print_info("\n[INFO]Testing the model on the hold-out test set...")
        # Get predictions
        predicted = clf.predict(x_test)
        if debug:
            # Print ROC curve and Clf report
            RocCurveDisplay.from_estimator(clf, x_test, y_test)
            metrics_report(y_test, predicted)
            # Plot normalized confusion matrix
            plot_confusion_matrix(y_test, predicted,
                                  classes=["Lminus", "Lplus"],
                                  normalize=True, title='Normalized confusion matrix')
        # Calculate the accuracy for each target/class and print
        scores_test["Acc"].append(acc_per_class(y_test, predicted))
        scores_test["ROC"].append(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
        # Get indexes of wrongly predicted traces
        diff = (predicted == y_test)
        test_diff_idx = np.where(diff == False)[0]

        for metric in ["Acc", "ROC"]:
            # Append the results to the df
            df_result.loc[dataset, (clf_name, metric)] = f'{mean(scores[metric]) * 100 :.2f}' + \
                                               '\xB1' + \
                                               f'{stdev(scores[metric]) * 100 :.2f}' + \
                                               f'({scores_test[metric][0] * 100 :.2f})'

    return df_result, test_diff_idx


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--seq_encoding", required=True, type=str,
        help="Possible encodings; 'one-hot' & 'n-gram' where n is an integer"
    )
    parser.add_argument(
        "--debug", default=False, action="store_true",
        help="When True, plots ROC-Curve & Confusion Matrix"
    )
    parser.add_argument(
        "--unique_traces", default=False, action="store_true",
        help="when True, duplicate traces(sequence of events) are removed from dataset"
    )
    parser.add_argument(
        "--remove_biased_feats", default=False, action="store_true",
        help="when True, the biased features are removed from dataset, e.g. patient is dead in COVID dataset"
    )
    parser.add_argument(
        "--split_by", required=True, type=str,
        help="Possible split options; 'unique', 'time', and 'random'."
    )
    return parser


if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()

    # Define constants
    DATA_DIR = "./data/"
    MODEL_DIR = "./models/"
    SAVE_DIR = MODEL_DIR + "without_biased" if args.remove_biased_feats else MODEL_DIR + "with_biased"

    DATASETS = ['BPIC17', 'BPIC18', 'Traffic', 'COVID', 'Hospital']
    MAX_SEQ_LEN = [40, 40, 15, 12, 40]  # max sequence length to be used in each dataset
    # Define the models
    MODELS = {"LogReg_L1": LogisticRegression(max_iter=500,
                                              penalty='l1',  # Regularization (Lasso)
                                              solver='liblinear',
                                              C=0.9),  # Regularization strength: the INVERSE of alpha, used by Lasso.}
              "DT": DecisionTreeClassifier(random_state=0),
              }

    # Construct the table where the scores will be appended
    col_multiindex = pd.MultiIndex.from_product([MODELS.keys(), ['Acc', 'ROC']])
    df_result = pd.DataFrame(0, columns=col_multiindex, index=DATASETS)

    for dataset, max_seq_len in zip(DATASETS, MAX_SEQ_LEN):
        # Load data and convert to pandas DataFrame
        (df_lm_train, df_lm_test,
         df_lp_train, df_lp_test) = load_split_data_as_df(data_dir=DATA_DIR, dataset=dataset, split_by=args.split_by)

        # Construct traces from both df, concat them and generate train data
        (x_train, x_test,
         y_train, y_test,
         X_raw, encoding) = construct_train_data(df_lm_train, df_lm_test,
                                                 df_lp_train, df_lp_test,
                                                 max_seq_len=max_seq_len,
                                                 seq_encoding=args.seq_encoding,
                                                 remove_duplicate_traces=args.unique_traces,
                                                 remove_biased_features=args.remove_biased_feats,
                                                 dataset=dataset)

        df_result, _ = pipeline(x_train, y_train, x_test, y_test, split_by=args.split_by, encoding=args.seq_encoding,
                                df_result=df_result, models=MODELS,
                                save_dir=SAVE_DIR, dataset=dataset,
                                n_splits=5, n_repeats=50, debug=args.debug)

    print(df_result)
    if args.remove_biased_feats:
        df_result.to_csv(f"./reports/results-not_biased-{args.seq_encoding}.csv")
    else:
        df_result.to_csv(f"./reports/results-with_biased-{args.seq_encoding}.csv")
