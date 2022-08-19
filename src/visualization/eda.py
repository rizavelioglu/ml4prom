import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style('dark')


def analyse_data(df_lm, df_lp,
                 xlim: list = None,
                 ylim: list = None,
                 percentiles: list = [.5, .75, .99]):
    """

    Parameters
    ----------
    percentiles
    df_lm
    df_lp
    xlim
    ylim

    Returns
    -------

    """
    # Maximum-Minimum length of a trace
    lm_trace_len = df_lm.groupby(by='case:concept:name').size()
    lp_trace_len = df_lp.groupby(by='case:concept:name').size()
    df_stats = pd.concat([lm_trace_len, lp_trace_len], axis=1, keys=['L-', 'L+'])
    print(df_stats.describe(percentiles=percentiles).astype(int))

    # Plot histogram of trace lengths
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.histplot(df_lm.groupby(by='case:concept:name').size(), kde=True,
                 color="red", stat='percent', discrete=True, ax=ax)

    sns.histplot(df_lp.groupby(by='case:concept:name').size(), kde=True,
                 color="blue", stat='percent', discrete=True, ax=ax)

    ax.set_title('Histogram of trace length')
    ax.set_xlabel('Trace Length')
    ax.set_ylabel('Probability')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(['L-', 'L+'], title='Logs')
    plt.show()

    # Unique event names
    print(f"Unique event names in:",
          f"\nLminus({df_lm['concept:name'].nunique()} events): \n{df_lm['concept:name'].unique()}",
          f"\n\nLplus({df_lp['concept:name'].nunique()} events): \n{df_lp['concept:name'].unique()}",
          f"\n\nTheir difference in terms of events:",
          f"\n{set(df_lp['concept:name']).symmetric_difference(set(df_lm['concept:name']))}")


def plot_unique_events_per_trace_location(lm_traces_train, lp_traces_train,
                                          lm_traces_test, lp_traces_test):
    """

    Parameters
    ----------
    lm_traces_test
    lp_traces_test
    lp_traces_train
    lm_traces_train

    Returns
    -------

    Examples:
        Plot the graph for all traces combined.
        >>> plot_unique_events_per_trace_location(lm_traces_train, lm_traces_test, lp_traces_train, lp_traces_test)

    """
    # Concat all traces to plot a single graph
    traces = pd.concat([lm_traces_train, lp_traces_train,
                        lm_traces_test, lp_traces_test],
                       axis='rows')
    # Expand pd.Series(traces) to a pd.DataFrame
    traces = traces.str.split(',', expand=True)

    fig, ax = plt.subplots(figsize=(12, 3))
    plt.bar(traces.nunique().index, traces.nunique().values)
    ax.set_title("#of unique events per location in all traces")
    ax.set_xlabel("Position")
    ax.set_ylabel("#of unique events")
    fig.tight_layout()
    plt.grid()
    plt.show()


def plot_cumulative_dist_of_traces(lm_traces_train, lm_traces_test,
                                   lp_traces_train, lp_traces_test,
                                   dataset, top_k: int = 10):
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(lm_traces_train.value_counts().values[:top_k]) / lm_traces_train.shape[0], label='L-')
    plt.plot(np.cumsum(lm_traces_test.value_counts().values[:top_k]) / lm_traces_test.shape[0], label='L- test')
    plt.plot(np.cumsum(lp_traces_train.value_counts().values[:top_k]) / lp_traces_train.shape[0], label="L+")
    plt.plot(np.cumsum(lp_traces_test.value_counts().values[:top_k]) / lp_traces_test.shape[0], label="L+ test")
    plt.title(f"{dataset}-Cumulative distribution of top-{top_k} most frequent traces")
    plt.xticks(np.arange(top_k), np.arange(1, top_k + 1))
    plt.xlabel("# of unique traces")
    plt.legend()
    plt.grid()


def plot_hist_by_attr(df_lm_train, df_lm_test,
                      df_lp_train, df_lp_test, attr: str, trace_case_id="case:concept:name"):
    plt.figure(figsize=(19, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.suptitle(f'Histogram of the `{attr}` attribute;')

    # Plot for train set
    df_lm_attr = df_lm_train[[trace_case_id, attr]].dropna().groupby(trace_case_id).agg(np.unique).astype(str).value_counts()
    df_lp_attr = df_lp_train[[trace_case_id, attr]].dropna().groupby(trace_case_id).agg(np.unique).astype(str).value_counts()

    df = pd.DataFrame({'L-': df_lm_attr, 'L+': df_lp_attr})
    ax = df.plot.bar(ax=ax1, rot=70, title='on train set')
    for container in ax.containers:
        ax.bar_label(container)

    # Plot for test set
    df_lm_attr = df_lm_test[[trace_case_id, attr]].dropna().groupby(trace_case_id).agg(np.unique).astype(str).value_counts()
    df_lp_attr = df_lp_test[[trace_case_id, attr]].dropna().groupby(trace_case_id).agg(np.unique).astype(str).value_counts()

    df = pd.DataFrame({'L-': df_lm_attr, 'L+': df_lp_attr})
    ax = df.plot.bar(ax=ax2, rot=70, title='on test set')
    for container in ax.containers:
        ax.bar_label(container)

    plt.show()
