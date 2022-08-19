import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
import numpy as np


def train_test_split(data_dir, dataset, trace_case_id="case:concept:name", test_size=.3,
                     split_by_unique_traces: bool = False,
                     split_by_time: bool = True):
    # Read .xes log files
    log_lm = pm4py.read_xes(f"{data_dir}/{dataset}/Lm_final.xes")
    log_lp = pm4py.read_xes(f"{data_dir}/{dataset}/Lp_final.xes")

    # Convert them to pandas df
    df_lm = pm4py.convert_to_dataframe(log_lm)
    df_lp = pm4py.convert_to_dataframe(log_lp)

    # Free up the RAM
    del(log_lm, log_lp)

    # Collect unique traces from lm & lp and do the split
    if split_by_unique_traces:
        # Get traces from event log(pandas df)
        lm_traces = df_lm[[trace_case_id, "concept:name"]].groupby(trace_case_id).agg(','.join)['concept:name']
        lp_traces = df_lp[[trace_case_id, "concept:name"]].groupby(trace_case_id).agg(','.join)['concept:name']

        # Get unique traces
        lm_traces_unique = lm_traces.reset_index().groupby("concept:name").agg(','.join)
        lp_traces_unique = lp_traces.reset_index().groupby("concept:name").agg(','.join)

        # Compute number of samples in train and test set
        n_traces_lm = lm_traces.nunique()
        n_traces_lp = lp_traces.nunique()

        n_traces_lm_train = int((1. - test_size) * n_traces_lm)  # Sample class wise - be aware of highly imbalanced data!
        n_traces_lp_train = int((1. - test_size) * n_traces_lp)

        # Sample from unique traces and add non-unique traces,too! (required for discovering a process model)
        lm_ids_train = ",".join(lm_traces_unique.sample(n=n_traces_lm_train, random_state=1)[trace_case_id]).split(",");lm_ids_test = list(set(df_lm[trace_case_id].unique()) - set(lm_ids_train))
        lp_ids_train = ",".join(lp_traces_unique.sample(n=n_traces_lp_train, random_state=1)[trace_case_id]).split(",");lp_ids_test = list(set(df_lp[trace_case_id].unique()) - set(lp_ids_train))

        # Split traces into train and test parts (dataframes)
        df_lm_train = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_train)].reset_index(drop=True);df_lm_test = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_test)].reset_index(drop=True)
        df_lp_train = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_train)].reset_index(drop=True);df_lp_test = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_test)].reset_index(drop=True)

    # Sort traces by time and take the first (1-test_size)% as train, the rest as test (uniqueness is not considered!)
    elif split_by_time:
        # Sort data by time and keep only the first event(w.r.t. time) in a trace
        lm_ids = df_lm.loc[~df_lm.sort_values('time:timestamp').duplicated(trace_case_id, keep='first')].sort_values('time:timestamp')[trace_case_id].values
        lp_ids = df_lp.loc[~df_lp.sort_values('time:timestamp').duplicated(trace_case_id, keep='first')].sort_values('time:timestamp')[trace_case_id].values

        # Compute number of samples in train and test set
        n_traces_lm = lm_ids.shape[0]
        n_traces_lp = lp_ids.shape[0]

        n_traces_lm_train = int((1. - test_size) * n_traces_lm)  # Sample class wise - be aware of highly imbalanced data!
        n_traces_lp_train = int((1. - test_size) * n_traces_lp)

        # Collect the ids for train & test sets
        lm_ids_train = lm_ids[:n_traces_lm_train];lm_ids_test = lm_ids[n_traces_lm_train:]
        lp_ids_train = lp_ids[:n_traces_lp_train];lp_ids_test = lp_ids[n_traces_lp_train:]

        # Construct train & test sets
        df_lm_train = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_train)].reset_index(drop=True);df_lm_test = df_lm.loc[df_lm[trace_case_id].isin(lm_ids_test)].reset_index(drop=True)
        df_lp_train = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_train)].reset_index(drop=True);df_lp_test = df_lp.loc[df_lp[trace_case_id].isin(lp_ids_test)].reset_index(drop=True)

    # Convert back to event log objects
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: trace_case_id}

    lm_train = log_converter.apply(df_lm_train, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG);lm_test = log_converter.apply(df_lm_test, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    lp_train = log_converter.apply(df_lp_train, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG);lp_test = log_converter.apply(df_lp_test, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    return lm_train, lp_train, lm_test, lp_test


def export_event_log_to_xes(event_log_data, file_out_name):
    xes_exporter.apply(event_log_data, file_out_name)


def evaluate_pm_model(pm, lp_test, lm_test, threshold = .6):  # TODO: Threshold  depends on the data set
    # Token replay (TODO: Other conformance checking methods are possible)
    results_lp = token_replay.apply(lp_test, *pm)
    results_lm = token_replay.apply(lm_test, *pm)

    # Extract fitness values
    fitness_lp = np.array([r["trace_fitness"] for r in results_lp])
    fitness_lm = np.array([r["trace_fitness"] for r in results_lm])

    # Perform a simple threshold based classification
    return np.sum(fitness_lp >= threshold) / (1. * len(fitness_lp)), np.sum(fitness_lm <= threshold) / (1. * len(fitness_lm))


if __name__ == "__main__":
    # Load and split data
    lm_train, lp_train, lm_test, lp_test = train_test_split("../data", "Traffic")

    # "Discover"/Estimate process model
    pm = alpha_miner.apply(lp_train)  # Estimate a process model (petri net)

    # Evaluate process model on test set
    print(evaluate_pm_model(pm, lp_test, lm_test))  # TODO: Estimated petri net does not perform that well as a classifier (in particular on BPIC18 it shows a pretty bad performance! - maybe a bad local optimum?!)
