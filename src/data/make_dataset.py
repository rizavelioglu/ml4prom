import argparse
import pm4py
import os
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from src.utils import print_info


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--split_by", required=True, type=str,
        help="Possible split options; 'unique', 'time', and 'random'."
    )
    parser.add_argument(
        "--test_size", default=.3, type=float,
        help="The percentage of the test dataset."
    )
    parser.add_argument(
        "--n_threshold", default=0, type=float,
        help="Keep traces only that occur more than `n_threshold` times."
    )
    return parser


def export_event_log_to_xes(event_log_data, file_out_name):
    xes_exporter.apply(event_log_data, file_out_name)


def load_xes_as_df(data_dir, dataset_name):
    # Read .xes log files
    log_lm = pm4py.read_xes(f"{data_dir}/{dataset_name}/Lm_final.xes")
    log_lp = pm4py.read_xes(f"{data_dir}/{dataset_name}/Lp_final.xes")
    # Convert them to pandas df
    df_lm = pm4py.convert_to_dataframe(log_lm)
    df_lp = pm4py.convert_to_dataframe(log_lp)

    return df_lm, df_lp


def train_test_split_unique(data_dir, dataset_name, test_size, n_threshold=0, trace_case_id="case:concept:name"):
    # Load datasets as pd.DataFrame
    df_lm, df_lp = load_xes_as_df(data_dir, dataset_name)
    # Get traces from event log(pandas df)
    lm_traces = df_lm[[trace_case_id, "concept:name"]].groupby(trace_case_id).agg(','.join)['concept:name']
    lp_traces = df_lp[[trace_case_id, "concept:name"]].groupby(trace_case_id).agg(','.join)['concept:name']

    # Keep only the traces which occur more than `n_threshold` times
    lm_traces = lm_traces[lm_traces.isin(lm_traces.value_counts().loc[lambda x: x > n_threshold].keys())]
    lp_traces = lp_traces[lp_traces.isin(lp_traces.value_counts().loc[lambda x: x > n_threshold].keys())]

    # Get unique traces
    lm_traces_unique = lm_traces.reset_index().groupby("concept:name").agg(','.join)
    lp_traces_unique = lp_traces.reset_index().groupby("concept:name").agg(','.join)

    # Compute number of samples in train and test set
    n_traces_lm = lm_traces_unique.size
    n_traces_lp = lp_traces_unique.size

    n_traces_lm_train = int((1. - test_size) * n_traces_lm)  # Sample class wise-be aware of highly imbalanced data!
    n_traces_lp_train = int((1. - test_size) * n_traces_lp)

    # Sample from unique traces and add non-unique traces,too! (required for discovering a process model)
    lm_ids_train = ",".join(lm_traces_unique.sample(n=n_traces_lm_train, random_state=42)[trace_case_id]).split(",")
    lp_ids_train = ",".join(lp_traces_unique.sample(n=n_traces_lp_train, random_state=42)[trace_case_id]).split(",")
    # Remove duplicate traces in test sets
    lm_traces_test = lm_traces[~lm_traces.index.isin(lm_ids_train)]
    lm_ids_test = lm_traces_test[~lm_traces_test.duplicated(keep='first')].index.tolist()
    lp_traces_test = lp_traces[~lp_traces.index.isin(lp_ids_train)]
    lp_ids_test = lp_traces_test[~lp_traces_test.duplicated(keep='first')].index.tolist()

    # Split traces into train and test parts (dataframes)
    df_lm_train = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_train)].reset_index(drop=True)
    df_lm_test = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_test)].reset_index(drop=True)
    df_lp_train = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_train)].reset_index(drop=True)
    df_lp_test = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_test)].reset_index(drop=True)

    return convert_split_data_to_event_log(df_lm_train, df_lm_test, df_lp_train, df_lp_test, trace_case_id)


def train_test_split_time(data_dir, dataset_name, test_size, trace_case_id="case:concept:name"):
    """Sort traces by time and take the first (1-test_size)% as train, the rest as test (uniqueness is not considered!).

    Parameters
    ----------
    data_dir
    dataset_name
    trace_case_id
    test_size

    Returns
    -------

    """
    # Load datasets as pd.DataFrame
    df_lm, df_lp = load_xes_as_df(data_dir, dataset_name)

    # Sort data by time and keep only the first event(w.r.t. time) in a trace
    lm_ids = df_lm.loc[~df_lm.sort_values('time:timestamp').duplicated(trace_case_id, keep='first')].sort_values(
        'time:timestamp')[trace_case_id].values
    lp_ids = df_lp.loc[~df_lp.sort_values('time:timestamp').duplicated(trace_case_id, keep='first')].sort_values(
        'time:timestamp')[trace_case_id].values

    # Compute number of samples in train and test set
    n_traces_lm = lm_ids.shape[0]
    n_traces_lp = lp_ids.shape[0]

    n_traces_lm_train = int((1. - test_size) * n_traces_lm)  # Sample class wise-be aware of highly imbalanced data!
    n_traces_lp_train = int((1. - test_size) * n_traces_lp)

    # Collect the ids for train & test sets
    lm_ids_train = lm_ids[:n_traces_lm_train]
    lm_ids_test = lm_ids[n_traces_lm_train:]
    lp_ids_train = lp_ids[:n_traces_lp_train]
    lp_ids_test = lp_ids[n_traces_lp_train:]

    # Construct train & test sets
    df_lm_train = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_train)].reset_index(drop=True)
    df_lm_test = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_test)].reset_index(drop=True)
    df_lp_train = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_train)].reset_index(drop=True)
    df_lp_test = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_test)].reset_index(drop=True)

    return convert_split_data_to_event_log(df_lm_train, df_lm_test, df_lp_train, df_lp_test, trace_case_id)


def train_test_split_random(data_dir, dataset_name, test_size, n_threshold=1, trace_case_id="case:concept:name"):
    """Split the data into train&test by random sampling (non-unique traces are preserved!).

    Parameters
    ----------
    n_threshold
    data_dir
    dataset_name
    trace_case_id
    test_size

    Returns
    -------

    """
    # Load datasets as pd.DataFrame
    df_lm, df_lp = load_xes_as_df(data_dir, dataset_name)

    # Get traces from event log(pandas df)
    lm_traces = df_lm[[trace_case_id, "concept:name"]].groupby(trace_case_id).agg(','.join)['concept:name']
    lp_traces = df_lp[[trace_case_id, "concept:name"]].groupby(trace_case_id).agg(','.join)['concept:name']

    # Keep only the traces which occur more than `n_threshold` times
    lm_traces = lm_traces[lm_traces.isin(lm_traces.value_counts().loc[lambda x: x > n_threshold].keys())]
    lp_traces = lp_traces[lp_traces.isin(lp_traces.value_counts().loc[lambda x: x > n_threshold].keys())]

    # Compute number of samples in train and test set
    n_traces_lm = lm_traces.shape[0]
    n_traces_lp = lp_traces.shape[0]

    n_traces_lm_train = int((1. - test_size) * n_traces_lm)  # Sample class wise-be aware of highly imbalanced data!
    n_traces_lp_train = int((1. - test_size) * n_traces_lp)

    # Sample from non-unique traces (all traces are required for discovering a process model)
    lm_ids_train = lm_traces.sample(n=n_traces_lm_train, random_state=42).index
    lm_ids_test = lm_traces[~lm_traces.index.isin(lm_ids_train)].index
    lp_ids_train = lp_traces.sample(n=n_traces_lp_train, random_state=42).index
    lp_ids_test = lp_traces[~lp_traces.index.isin(lp_ids_train)].index

    # Construct train & test sets
    df_lm_train = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_train)].reset_index(drop=True)
    df_lm_test = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_test)].reset_index(drop=True)
    df_lp_train = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_train)].reset_index(drop=True)
    df_lp_test = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_test)].reset_index(drop=True)

    return convert_split_data_to_event_log(df_lm_train, df_lm_test, df_lp_train, df_lp_test, trace_case_id)


def convert_split_data_to_event_log(df_lm_train, df_lm_test,
                                    df_lp_train, df_lp_test,
                                    trace_case_id):
    # Convert back to event log objects
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: trace_case_id}

    lm_train = log_converter.apply(df_lm_train, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    lm_test = log_converter.apply(df_lm_test, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    lp_train = log_converter.apply(df_lp_train, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    lp_test = log_converter.apply(df_lp_test, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    return lm_train, lm_test, lp_train, lp_test


if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()

    DATA_DIR = "./data"
    DATASETS = ['BPIC17', 'BPIC18', 'Traffic', 'COVID', 'Hospital']
    SPLIT_DATA_DIR = DATA_DIR + f'/train_test_splits_{args.split_by}/'
    SPLIT_METHODS = {"unique": train_test_split_unique,
                     "unique_v2": train_test_split_unique,
                     "time": train_test_split_time,
                     "random": train_test_split_random}

    os.makedirs(SPLIT_DATA_DIR + 'train')
    os.makedirs(SPLIT_DATA_DIR + 'test')

    for dataset in DATASETS:
        if os.path.exists(f"./data/{dataset}") is False:
            print_info(f"Skipping {dataset} dataset as it doesn't exist.")
            continue
        print_info(f"[INFO]Loading '{dataset}' and splitting it into train and test sets...")
        # Call respective function specified by `args.split_by` to collect split data as event log
        lm_train, lm_test, lp_train, lp_test = SPLIT_METHODS[args.split_by](DATA_DIR, dataset,
                                                                            args.test_size, args.n_threshold)
        # Save the logs in .xes format
        print_info(f"[INFO]Saving the split data to: {SPLIT_DATA_DIR}...")
        export_event_log_to_xes(lm_train, SPLIT_DATA_DIR + f'train/{dataset}-lm_train.xes')
        export_event_log_to_xes(lm_test, SPLIT_DATA_DIR + f'test/{dataset}-lm_test.xes')
        export_event_log_to_xes(lp_train, SPLIT_DATA_DIR + f'train/{dataset}-lp_train.xes')
        export_event_log_to_xes(lp_test, SPLIT_DATA_DIR + f'test/{dataset}-lp_test.xes')
