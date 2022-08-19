import pandas as pd
import numpy as np
import pm4py
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import utils
from nltk import ngrams
from src.utils import print_info, print_warning


def load_split_data_as_df(data_dir, dataset, split_by: str):
    """Loads and returns split dataset as pd.DataFrame

    Parameters
    ----------
    split_by : str
        Should be set to 'unique' if data is split randomly using unique traces
        Should be set to 'time' if data is split temporally
        Should be set to 'random' if data is split randomly
    dataset : str
        Name of the dataset in ['BPIC17', 'BPIC18', 'COVID', 'Hospital', 'Traffic']
    data_dir : str
        The path to the data folder

    Returns
    -------
    df_lm_train : pd.DataFrame
        Training dataset of Lminus (lm)
    df_lm_test : pd.DataFrame
        Test dataset of Lminus (lm)
    df_lp_train : pd.DataFrame
        Training dataset of Lplus (lp)
    df_lp_test : pd.DataFrame
        Test dataset of Lplus (lp)
    """
    print_info(f"[INFO]Loading '{dataset}' dataset from '{data_dir}train_test_splits_{split_by}'...")
    # Read train & test sets
    log_lm_train = pm4py.read_xes(f"{data_dir}train_test_splits_{split_by}/train/{dataset}-lm_train.xes")
    log_lm_test = pm4py.read_xes(f"{data_dir}train_test_splits_{split_by}/test/{dataset}-lm_test.xes")
    log_lp_train = pm4py.read_xes(f"{data_dir}train_test_splits_{split_by}/train/{dataset}-lp_train.xes")
    log_lp_test = pm4py.read_xes(f"{data_dir}train_test_splits_{split_by}/test/{dataset}-lp_test.xes")

    # Convert them to pandas df
    df_lm_train = pm4py.convert_to_dataframe(log_lm_train)
    df_lm_test = pm4py.convert_to_dataframe(log_lm_test)
    df_lp_train = pm4py.convert_to_dataframe(log_lp_train)
    df_lp_test = pm4py.convert_to_dataframe(log_lp_test)

    return df_lm_train, df_lm_test, df_lp_train, df_lp_test


def construct_train_data(df_lm_train, df_lm_test,
                         df_lp_train, df_lp_test,
                         max_seq_len: int = 50, seq_encoding: str = 'one-hot',
                         remove_duplicate_traces: bool = False, remove_biased_features: bool = False,
                         return_traces: bool = False, dataset=None,
                         ):
    if remove_biased_features:
        lm_traces_train, lp_traces_train = _remove_events_from_trace(df_lm_train, df_lp_train, dataset)
        lm_traces_test, lp_traces_test = _remove_events_from_trace(df_lm_test, df_lp_test, dataset)
    else:
        # Group by event names to get traces
        lm_traces_train = df_lm_train[['concept:name', "case:concept:name"]].groupby('case:concept:name').agg(','.join)[
            'concept:name'].str.split(',', expand=True)
        lp_traces_train = df_lp_train[['concept:name', "case:concept:name"]].groupby('case:concept:name').agg(','.join)[
            'concept:name'].str.split(',', expand=True)
        lm_traces_test = df_lm_test[['concept:name', "case:concept:name"]].groupby('case:concept:name').agg(','.join)[
            'concept:name'].str.split(',', expand=True)
        lp_traces_test = df_lp_test[['concept:name', "case:concept:name"]].groupby('case:concept:name').agg(','.join)[
            'concept:name'].str.split(',', expand=True)

    if remove_duplicate_traces:
        # Get duplicate indexes
        idx_lm_train = lm_traces_train[lm_traces_train.duplicated(keep='first')].index
        idx_lp_train = lp_traces_train[lp_traces_train.duplicated(keep='first')].index
        idx_lm_test = lm_traces_test[lm_traces_test.duplicated(keep='first')].index
        idx_lp_test = lp_traces_test[lp_traces_test.duplicated(keep='first')].index
        # Remove duplicates
        lm_traces_train = lm_traces_train[~lm_traces_train.index.isin(idx_lm_train)]
        lp_traces_train = lp_traces_train[~lp_traces_train.index.isin(idx_lp_train)]
        lm_traces_test = lm_traces_test[~lm_traces_test.index.isin(idx_lm_test)]
        lp_traces_test = lp_traces_test[~lp_traces_test.index.isin(idx_lp_test)]

    # concat all the traces
    X_raw = pd.concat([lm_traces_train, lp_traces_train,
                       lm_traces_test, lp_traces_test],
                      axis='rows')
    # Take the first 'max_seq_len' events
    X_raw = X_raw.loc[:, :max_seq_len]

    # One-hot encode the event names
    X, encoding = _encode_features(X_raw, encoding=seq_encoding)

    # Construct labels for both desirable(lp) & undesirable traces(lm)
    y_lm_train = np.zeros((lm_traces_train.shape[0],))
    y_lp_train = np.ones((lp_traces_train.shape[0],))
    y_lm_test = np.zeros((lm_traces_test.shape[0],))
    y_lp_test = np.ones((lp_traces_test.shape[0],))

    y_train = np.concatenate((y_lm_train, y_lp_train))
    y_test = np.concatenate((y_lm_test, y_lp_test))

    X_train = X[:y_train.shape[0]]
    X_test = X[y_train.shape[0]:]

    # Shuffle train dataset
    X_train, y_train = utils.shuffle(X_train, y_train)

    if return_traces:
        return X_train, X_test, y_train, y_test, X_raw, encoding, \
               lm_traces_train, lm_traces_test, lp_traces_train, lp_traces_test
    else:
        return X_train, X_test, y_train, y_test, X_raw, encoding


def _remove_events_from_trace(df_lm, df_lp, dataset):
    """Remove specific events/actions from traces/sequences.

    Some events reveal the label of the trace, e.g. 'Discharge dead' in COVID dataset means the patient is dead. With
    such events the task of classification becomes too simple for ML models. This function removes such 'biased' events
    from traces to make the task more challenging.

    Parameters
    ----------
    df_lm : pd.DataFrame
        The event log belonging to the negative class (undesirable traces)
    df_lp : pd.DataFrame
        The event log belonging to the positive class (desirable traces)
    dataset : str
        The name of the dataset

    Returns
    -------

    """
    biased_features = {'BPIC17': {'Lminus': [['A_Incomplete']],
                                  'Lplus': [None]},
                       'BPIC18': {'Lminus': [None],
                                  'Lplus': [None]},
                       'Traffic': {'Lminus': [['Send for Credit Collection']],
                                   'Lplus': [None]},
                       'COVID': {'Lminus': [['Discharge dead']],
                                 'Lplus': [['Discharge alive']]},
                       'Hospital': {'Lminus': [None],
                                    'Lplus': [['ZDBC_BEHAN']]}
                       }

    # Group by event id's to get traces
    df_lm_traces = df_lm[['concept:name', "case:concept:name"]].groupby('case:concept:name').agg(','.join)[
        'concept:name']
    df_lp_traces = df_lp[['concept:name', "case:concept:name"]].groupby('case:concept:name').agg(','.join)[
        'concept:name']

    # To remove the bias in 'Traffic' dataset, remove the informative feature (Payment)
    # The event 'Payment' is removed from lp_traces such that all traces' last event is different from Payment
    if dataset == 'Traffic':
        df_lp_traces = df_lp_traces.str.replace(r'(,Payment)+$', '', regex=True)

    for feat in biased_features[dataset]['Lminus']:
        if feat:
            df_lm_traces = df_lm_traces.str.replace(f',{feat[0]}', '', regex=True)
    # Split each event to respective columns
    df_lm_traces = df_lm_traces.str.split(',', expand=True)

    for feat in biased_features[dataset]['Lplus']:
        if feat:
            df_lp_traces = df_lp_traces.str.replace(f',{feat[0]}', '', regex=True)
    # Split each event to respective columns
    df_lp_traces = df_lp_traces.str.split(',', expand=True)

    return df_lm_traces, df_lp_traces


def _encode_features(data, encoding):
    """Encode the given data with respect to the given 'encoding' technique.

    Parameters
    ----------
    data : pd.DataFrame()
        The raw data to be encoded.
    encoding : str
        The encoding technique to be used to encode the input data. Possible techniques
        are; 'one-hot', and n-grams, e.g. '1-gram', '2-gram', '3-gram', etc.

    Returns
    -------
    np.array()
        The encoded input data.
    """
    if encoding == 'one-hot':
        # One-hot encode the event names
        dummies = pd.get_dummies(data)
        # Convert pd.DataFrame to numpy
        data_encoded = dummies.to_numpy()
        return data_encoded, dummies

    elif encoding.__contains__('-gram'):
        # Get the 'n' of n-gram
        n = int(encoding.split('-')[0])

        # Pad input sequences with 0's
        data = data.fillna('0')
        # Join all events(str) into a single sentence
        data = data.loc[:, pd.RangeIndex(0, data.shape[1])].agg(','.join, axis=1)
        # Add 'start' and 'end' tokens
        data = '<start>,' + data.astype(str) + ',0'
        data = data.apply(lambda x: re.sub(r"(?<=[a-zA-Z].),0", ",<end>", x))

        # Create n-gram lookup table that looks like;
        # index, ngram, count --> e.g. 3, (action1,action2,action3), 125
        mapping = dict()
        for row in data:
            for ngram in ngrams(row.split(','), n):
                if ngram not in mapping.keys():
                    mapping[ngram] = 1
                else:
                    mapping[ngram] += 1
        # Convert mapping to DataFrame
        mapping = pd.DataFrame.from_dict(mapping,
                                         orient='index',
                                         columns=['count']).rename_axis('ngram')
        # Sort by ngram counts
        mapping = mapping.sort_values('count', ascending=False).reset_index()

        # # Encode sequences using their int indexes (replace ngrams by their index in mapping);
        # # (action1,action2,action3) will be replaced with 3
        # data_encoded = list()
        # for row in data:
        #     encoded_seq = list()
        #     for ngram in ngrams(row.split(','), n):
        #         encoded_seq.append(mapping[mapping.ngram == ngram].index[0])
        #     data_encoded.append(encoded_seq)
        # # Convert the encoded data to np.array
        # data_encoded = np.array(data_encoded)

        # Encode sequences in one-hot fashion
        data_encoded = pd.DataFrame(columns=mapping.ngram.values,
                                    index=data.index)

        for idx, ngram in enumerate(data_encoded.columns.values):
            data_encoded.iloc[:, idx] = data.str.contains(','.join(ngram), regex=True)
        data_encoded = data_encoded.astype(int)

        return data_encoded.to_numpy(), mapping

    # Map each event to a unique number
    elif encoding == 'integer':
        # Join activities for each case
        data = data.stack().groupby(level=0).agg(','.join).reindex_like(data)

        # Get unique events in the event log
        activities = set()
        _ = data.str.split(",").apply(activities.update)
        # Create the integer mapping of activities
        act_ids = range(1, len(activities) + 1)
        act_to_int = dict(zip(activities, act_ids))
        # Add the padding token
        act_to_int["[PAD]"] = 0

        # Replace activities with unique integers
        X = list()
        for seq in data.values:
            X.append([act_to_int[act] for act in seq.split(',')])

        X = pad_sequences(X, padding="post")
        X = np.array(X, dtype=np.float32)

        return X, act_to_int

    else:
        return "encoding technique must be either 'one-hot', 'integer' or 'n-gram'!"
