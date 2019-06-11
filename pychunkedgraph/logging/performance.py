import datetime
import numpy as np
import matplotlib as mpl

try:
    mpl.use('Agg')
except:
    pass

from matplotlib import pyplot as plt

from pychunkedgraph.logging import flask_log_db
from pychunkedgraph.backend import chunkedgraph
from google.cloud import datastore
from google.auth import credentials, default as default_creds


def readout_log_db(table_id, filters, cols,
                   date_filter=datetime.datetime(year=2019, day=30, month=3)):
    if date_filter.tzinfo is None:
        date_filter = chunkedgraph.UTC.localize(date_filter)

    credentials, project_id = default_creds()
    client = datastore.Client(project=project_id, credentials=credentials)
    log_db = flask_log_db.FlaskLogDatabase(table_id, client=client)

    query = log_db.client.query(kind=log_db.kind, namespace=log_db.namespace)

    for filter_ in filters:
        query.add_filter(*filter_)

    data = []
    query_iter = query.fetch()
    for e in query_iter:
        if e["date"] > date_filter:
            col_data = []
            for col in cols:
                col_data.append(e[col])

            data.append(col_data)

        # if len(data) > 10000:
        #     break

    return data


def make_performance_plot(table_id, request_type, path):
    data = readout_log_db(table_id=table_id,
                          filters=[["request_type", "=", request_type]],
                          cols=["response_time(ms)"])

    data = np.array(data).squeeze()

    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor="white")
    ax1.tick_params(length=8, width=1.5, labelsize=20)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)

    ax2 = ax1.twinx()
    ax2.tick_params(length=8, width=1.5, labelsize=20)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)

    xmax = np.percentile(data, 98)
    nbins = int(np.max(data)/xmax * 25)

    hist_weights = np.ones_like(data) / float(len(data))
    values, base, _ = ax1.hist(data, bins=nbins, color=".4",
                               weights=hist_weights, linewidth=1, edgecolor="k")
    cumulative = np.cumsum(values) / np.sum(values)

    ax2.plot(base[:-1], cumulative, c=".2", lw=3)

    ax2.vlines([np.median(data)], 0, 1, colors=[0.7, 0.3, 0.3])
    ax2.text(np.median(data) - 0.02, 0.01, "Median", fontsize=18, rotation="vertical",
             verticalalignment="bottom", horizontalalignment="right", color=[0.7, 0.3, 0.3])
    ax2.vlines([np.percentile(data, 95)], 0, .5, colors=[0.7, 0.3, 0.3], linestyles="--")
    ax2.text(np.percentile(data, 95) - 0.02, 0.01, "95th percentile", fontsize=18, rotation="vertical",
             verticalalignment="bottom", horizontalalignment="right", color=[0.7, 0.3, 0.3])

    text = f"N = {len(data)}\nMedian = {np.median(data):.1f}ms\nMean = {np.mean(data):.2f}ms"
    # text = f"N = {len(data)}\nMedian = {np.median(data):.2f}ms\nMean = {np.mean(data):.2f}ms\n95th percentile = {np.percentile(data, 95):.2f}ms"

    ax2.text(xmax * .98, .55, text, fontsize=20, horizontalalignment="right")

    ax2.set_xlim(0, xmax)
    ax2.set_ylim(0, 1)
    ax1.set_ylim(0, values.max() * 1.05)

    ax2.set_ylabel("CDF", fontsize=22)
    ax1.set_ylabel("Normed count", fontsize=22)
    ax1.set_xlabel("Response time (ms)", fontsize=22)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_performance_plots(table_id, plot_dir):
    for request_type in ["split", "merge", "root", "leaves"]:
    # for request_type in ["split", "merge", "root"]:
        path = f"{plot_dir}/{request_type}.png"
        make_performance_plot(table_id, request_type, path)