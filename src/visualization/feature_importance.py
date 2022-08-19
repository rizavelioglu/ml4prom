from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import time
import joblib
import numpy as np
import pandas as pd
import argparse
from src.preprocessing import load_split_data_as_df, construct_train_data
from src.utils import print_info


# Feature Importance
def model_coef(model_dir: str, encoding, top_k: int = 20) -> None:
    # Load pre-trained model
    model = joblib.load(model_dir)

    # Get model coefficients
    coef = model.coef_.copy()
    coef = coef.reshape(coef.shape[1], )
    # Plot coefficients
    plt.figure(figsize=(25, 5))
    # plot feature importance
    plt.bar(list(range(len(coef))), coef)
    plt.title("LogisticRegression model coefficients")
    plt.xlabel("Features")
    plt.ylabel("model.coef_")
    plt.grid()
    plt.show()

    # Print most important features for both classes (Lminus & Lplus)
    importances = pd.DataFrame(coef, columns=["importance"])
    # Get feature names
    importances['Feature'] = encoding.ngram.values if '-gram' in model_dir else encoding.columns.values

    # Take the first `top_k` features sorted by absolute coefficient values
    top_k_idx = np.abs(importances.importance).sort_values()[-top_k:].index
    top_k_feats = importances.loc[top_k_idx].sort_values('importance')
    # Take the first (top_k/2) features from both positive and negative coefficients
    # top_k_feats = importances.sort_values(by="importance").iloc[np.r_[0:top_k/2, -top_k/2:0]]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.barh(top_k_feats['Feature'].astype(str), top_k_feats['importance'])
    ax.set_title(f"LogisticRegression model coefficients (top-{top_k} absolute coefficients)")
    fig.tight_layout()
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(linewidth=.3, linestyle="-.")
    plt.show()


def tree_mdi(model_dir: str, encoding, top_k: int = 20) -> None:
    """Feature importance based on Mean Decrease in Impurity(MDI)

    Parameters
    ----------
    model_dir
    encoding
    top_k

    Returns
    -------

    """
    # Get feature names (either one-hot or n-gram encoded)
    feat_names = encoding.ngram if '-gram' in model_dir else encoding.columns
    # Load pre-trained model
    model = joblib.load(model_dir)

    # Get model coefficients
    feature_importance = model.feature_importances_
    # Plot most important features for both classes (absolute values)
    sorted_idx = np.argsort(feature_importance)[-top_k:]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.barh(feat_names[sorted_idx].astype(str), feature_importance[sorted_idx])
    ax.set_title(f"Top-{top_k} feature importance based on Mean Decrease in Impurity(MDI)")
    ax.set_xlabel("feature_importance_")
    ax.set_ylabel("Features")
    fig.tight_layout()
    plt.show()


def tree_permutation(model_dir: str, x, y,
                     encoding, top_k: int = 20) -> None:
    """Feature importance based on Permutation Feature Importance.

    Source: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#tree-s-feature-importance-from-mean-decrease-in-impurity-mdi

    Parameters
    ----------
    y
    x
    model_dir
    encoding
    top_k

    Returns
    -------

    """
    # Get feature names (either one-hot or n-gram encoded)
    feat_names = encoding.ngram if '-gram' in model_dir else encoding.columns
    # Load pre-trained model
    model = joblib.load(model_dir)
    start_time = time.time()
    result = permutation_importance(model, x, y, n_repeats=10, random_state=42, n_jobs=2)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importance: "
          f"{elapsed_time:.3f} seconds")

    first_k = result.importances_mean.argsort()[-top_k:]
    last_k = result.importances_mean.argsort()[:top_k]
    sorted_idx = np.concatenate((last_k, first_k))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=feat_names[sorted_idx])
    ax.set_title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.grid()
    plt.show()


def tree_mdi_permutation(model_dir: str, x, y,
                         encoding, top_k: int = 20) -> None:
    """Plot Mean Decrease in Impurity(MDI, a.k.a.Gini Index) and Permutation Feature Importance for a given tree-based model.

    Parameters
    ----------
    model_dir
    x
    y
    encoding
    top_k

    Notes
    -------
    Check out [1] for more information on Permutation Feature Importance.

    [1]: https://christophm.github.io/interpretable-ml-book/feature-importance.html
    """
    # Get feature names (either one-hot or n-gram encoded)
    feat_names = encoding.ngram if '-gram' in model_dir else encoding.columns
    # Load pre-trained model
    model = joblib.load(model_dir)
    # Construct the plot
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"{str(model).split('(')[0]}", fontsize=14, color='red')

    # Calculate MDI
    feature_importance = model.feature_importances_
    # Get `top_k` features with the highest MDI value
    sorted_idx = np.argsort(feature_importance)[-top_k:]
    # Plot MDI
    plt.subplot(1, 2, 1)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.grid(linewidth=.3, linestyle="-.")
    plt.yticks(pos, feat_names[sorted_idx])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16, ticks=np.arange(0, 1, step=0.2))
    plt.title("Feature Importance (MDI)")

    # Calculate Permutation Feature Importance
    result = permutation_importance(model, x, y, scoring='roc_auc', n_repeats=50, random_state=42, n_jobs=20)
    # Get `top_k` features with the highest mean value
    sorted_idx = result.importances_mean.argsort()[-top_k:]
    # Plot Permutation Feature Importance
    plt.subplot(1, 2, 2)
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=feat_names[sorted_idx])
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.grid(linewidth=.3, linestyle="-.")
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.show()


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--dataset", required=True, type=str,
        help="The name of the dataset, e.g. BPIC17, COVID, etc."
    )
    parser.add_argument(
        "--split_by", required=True, type=str,
        help="The way dataset was split, e.g. unique, time, etc."
    )
    parser.add_argument(
        "--remove_biased_feats", default=False, action="store_true",
        help="when True, the biased features are removed from dataset, e.g. patient is dead in COVID dataset"
    )
    parser.add_argument(
        "--data_dir", required=True, type=str,
        help="Full path to dataset."
    )
    parser.add_argument(
        "--model_dir", required=True, type=str,
        help="Full path to trained models."
    )
    parser.add_argument(
        "--top_k", required=False, type=int, default=20,
        help="The number of most important features to plot"
    )
    return parser


if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()

    ENCODING_TYPES = ['1-gram', '2-gram', '3-gram']
    # sequences longer than value will be cropped
    MAX_SEQ_LEN = {"BPIC17": 40, "BPIC18": 40, "Traffic": 15, "COVID": 12, "Hospital": 40}
    MODEL_ROOT_DIR = args.model_dir + "without_biased" if args.remove_biased_feats else args.model_dir + "with_biased"

    # Load data and convert to pandas DataFrame
    df_lm_train, df_lm_test, df_lp_train, df_lp_test = load_split_data_as_df(data_dir=args.data_dir,
                                                                             dataset=args.dataset,
                                                                             split_by=args.split_by)
    # Plot feature importance
    for seq_encoding in ENCODING_TYPES:
        print_info(f"\nEncoding: {seq_encoding}\n")
        # Construct traces from both df, concat them and generate train data
        (X_train, X_test, y_train, y_test,
         X_raw, encoding) = construct_train_data(df_lm_train, df_lm_test,
                                                 df_lp_train, df_lp_test,
                                                 max_seq_len=MAX_SEQ_LEN[args.dataset],
                                                 seq_encoding=seq_encoding,
                                                 remove_duplicate_traces=False,
                                                 remove_biased_features=args.remove_biased_feats,
                                                 dataset=args.dataset)
        # Plot LogisticRegression model coefficients
        model_coef(f"{MODEL_ROOT_DIR}/{args.dataset}_{args.split_by}_LogisticRegression_{seq_encoding}.joblib",
                   encoding=encoding, top_k=args.top_k)
        # Plot Tree-based models' feature importance
        tree_mdi_permutation(f"{MODEL_ROOT_DIR}/{args.dataset}_{args.split_by}_DecisionTreeClassifier_{seq_encoding}.joblib",
                             x=X_test, y=y_test, encoding=encoding, top_k=args.top_k)
        tree_mdi_permutation(f"{MODEL_ROOT_DIR}/{args.dataset}_{args.split_by}_RandomForestClassifier_{seq_encoding}.joblib",
                             x=X_test, y=y_test, encoding=encoding, top_k=args.top_k)
        tree_mdi_permutation(f"{MODEL_ROOT_DIR}/{args.dataset}_{args.split_by}_XGBClassifier_{seq_encoding}.joblib",
                             x=X_test, y=y_test, encoding=encoding, top_k=args.top_k)
