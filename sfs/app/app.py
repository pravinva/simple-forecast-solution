# vim: set fdm=indent:
'''
 ___ _            _             
/ __(_)_ __  _ __| |___         
\__ \ | '  \| '_ \ / -_)        
|___/_|_|_|_| .__/_\___|        
 ___        |_|            _    
| __|__ _ _ ___ __ __ _ __| |_  
| _/ _ \ '_/ -_) _/ _` (_-<  _| 
|_|\___/_| \___\__\__,_/__/\__| 
 ___      _      _   _          
/ __| ___| |_  _| |_(_)___ _ _  
\__ \/ _ \ | || |  _| / _ \ ' \ 
|___/\___/_|\_,_|\__|_\___/_||_|

https://github.com/aws-samples/simple-forecat-solution/

USAGE:
    streamlit run ./app.py
    streamlit run -- ./app.py --local-dir LOCAL_DIR
                                 
'''
import os
import sys
import io
import glob
import time
import datetime
import base64
import pathlib
import textwrap
import argparse
import re
import json
import logging
import gzip

import boto3
import numpy as np
import pandas as pd
import awswrangler as wr
import streamlit as st
import plotly.express as pex
import plotly.graph_objects as go
import cloudpickle
import gzip

from collections import OrderedDict, deque, namedtuple
from concurrent import futures
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from sspipe import p, px
from streamlit import session_state as state
from stqdm import stqdm
from sfs import (load_data, resample, run_pipeline, run_cv_select,
    calc_smape,
    make_demand_classification, process_forecasts, make_perf_summary,
    make_health_summary, GROUP_COLS, EXP_COLS)

from lambdamap import LambdaExecutor, LambdaFunction
from awswrangler.exceptions import NoFilesFound
from streamlit import caching
from streamlit.uploaded_file_manager import UploadedFile
from streamlit.script_runner import RerunException
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from humanfriendly import format_timespan


ST_STATIC_PATH = pathlib.Path(st.__path__[0]).joinpath("static")
ST_DOWNLOADS_PATH = ST_STATIC_PATH.joinpath("downloads")
LAMBDAMAP_FUNC = "SfsLambdaMapFunction"
LOCAL_DIR = "/home/ec2-user/SageMaker"

if not os.path.exists(ST_DOWNLOADS_PATH):
    ST_DOWNLOADS_PATH.mkdir()

FREQ_MAP = OrderedDict(Daily="D", Weekly="W-MON", Monthly="MS")
FREQ_MAP_AFC = OrderedDict(Daily="D", Weekly="W", Monthly="M")
FREQ_MAP_LONG = {
    "D": "Daily", "W-MON": "Weekly", "W": "Weekly", "M": "Monthly",
    "MS": "Monthly"
}

FREQ_MAP_PD = {
    "D": "D",
    "W": "W-MON",
    "W-SUN": "W-MON",
    "W-MON": "W-MON",
    "M": "MS",
    "MS": "MS"
}


def validate(df):
    """Validate a dataset.
    """

    err_msgs = []
    warn_msgs = []

    # check column names
    for col in EXP_COLS:
        if col not in df:
            err_msgs.append(f"missing **{col}** column")

    msgs = {
        "errors": err_msgs,
        "warnings": warn_msgs
    }

    is_valid_file = len(err_msgs) == 0

    return df, msgs, is_valid_file


@st.cache
def load_file(path):
    """
    """

    if path.endswith(".csv.gz"):
        compression = "gzip"
    elif path.endswith(".csv"):
        compression = None
    else:
        raise NotImplementedError

    return pd.read_csv(path, dtype={"timestamp": str}, compression=compression)


class StreamlitExecutor(LambdaExecutor):
    """Custom LambdaExecutor to display a progress bar in the app.
    """

    def map(self, func, payloads, local_mode=False):
        """
        """

        if local_mode:
            f = func
        else:
            f = LambdaFunction(func, self._client, self._lambda_arn)
        
        ex = self._executor
        wait_for = [ex.submit(f, *p["args"], **p["kwargs"]) for p in payloads]

        return wait_for


def display_progress(wait_for, desc=None):
    """
    """

    # display progress of the futures
    pbar = stqdm(desc=desc, total=len(wait_for))
    prev_n_done = 0
    n_done = sum(f.done() for f in wait_for)

    while n_done != len(wait_for):
        diff = n_done - prev_n_done
        pbar.update(diff)
        prev_n_done = n_done
        n_done = sum(f.done() for f in wait_for)
        time.sleep(0.5)

    diff = n_done - prev_n_done
    pbar.update(diff)

    return


def run_lambdamap(df, horiz, freq):
    """
    """

    payloads = []

    freq = FREQ_MAP_PD[freq]
    df2 = load_data(df, freq)

    groups = df2.groupby(GROUP_COLS, as_index=False, sort=False)

    # generate payload
    for _, dd in groups:
        payloads.append(
            {"args": (dd, horiz, freq),
             "kwargs": {"obj_metric": "smape_mean", "cv_stride": 2}})

    executor = StreamlitExecutor(max_workers=min(1000, len(payloads)),
                                 lambda_arn=LAMBDAMAP_FUNC)
    wait_for = executor.map(run_cv_select, payloads)

    return wait_for


def display_ag_grid(df, auto_height=False, paginate=False,
    comma_cols=None, selection_mode=None, use_checkbox=False):
    """

    Parameters
    ----------
    df : pd.DataFrame
    auto_height : bool
    pagination : bool
    comma_cols : tuple or list
        Numeric columns to apply comma thousands separator.

    """

    gb = GridOptionsBuilder.from_dataframe(df)
    #gb.configure_selection("single")
    gb.configure_auto_height(auto_height)
    gb.configure_pagination(enabled=paginate)

    if selection_mode is not None:
        gb.configure_selection(selection_mode=selection_mode,
            use_checkbox=use_checkbox)

    comma_renderer = JsCode(textwrap.dedent("""
    function(params) {
        return  params.value
                      .toString()
                      .split( /(?=(?:\d{3})+(?:\.|$))/g ).join( "," )
    }
    """))

    for col in comma_cols:
        gb.configure_column(col, cellRenderer=comma_renderer)

    response = AgGrid(df, gridOptions=gb.build(), allow_unsafe_jscode=True)

    return response


def valid_launch_freqs():
    data_freq = state.report["data"]["freq"]
    valid_freqs = ["D", "W", "M"]

    if data_freq in ("D",):
        # don't allow daily forecasting yet
        valid_freqs = valid_freqs[1:]
    elif data_freq in ("W","W-MON",):
        valid_freqs = valid_freqs[1:]
    elif data_freq in ("M","MS",):
        valid_freqs = valid_freqs[2:]
    else:
        raise NotImplementedError

    return valid_freqs


def create_presigned_url(s3_path, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    parsed_url = urlparse(s3_path, allow_fragments=False)
    bucket_name = parsed_url.netloc
    object_name = parsed_url.path.strip("/")

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response


def make_df_backtests(df_results, parallel=False):
    """Expand df_results to a "long" dataframe with the columns:
    channel, family, item_id, timestamp, actual, backtest.

    """

    def _expand(dd):
        ts = np.hstack(dd["ts_cv"].apply(np.hstack))
        ys = np.hstack(dd["y_cv"].apply(np.hstack))
        yp = np.hstack(dd["yp_cv"].apply(np.hstack))
        df = pd.DataFrame({"timestamp": ts, "demand": ys, "backtest": yp})

        return df

    groups = df_results.query("rank == 1") \
                       .groupby(["channel", "family", "item_id"],
                                as_index=True, sort=False)
    
    if parallel:
        df_backtests = groups.parallel_apply(_expand)
    else:
        df_backtests = groups.apply(_expand)

    df_backtests["timestamp"] = pd.DatetimeIndex(df_backtests["timestamp"])

    return df_backtests.reset_index(["channel", "family", "item_id"])


def save_report():
    """
    """

    if "report" not in state:
        return

    if "path" not in state["report"]["data"]:
        st.warning(textwrap.dedent(f"""
        Warning: unable to save report, no input data was loaded.
        """))
        return

    with st.spinner("Saving Report ..."):
        now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        basename = os.path.basename(state["report"]["data"]["path"])
        local_path = f'/tmp/{basename}_{now_str}_sfs-report.pkl.gz'

        # save the report locally
        cloudpickle.dump(state["report"], gzip.open(local_path, "wb"))

        # upload the report to s3
        s3_path = \
            f'{state["report"]["sfs"]["s3_sfs_reports_path"]}/{os.path.basename(local_path)}'

        parsed_url = urlparse(s3_path, allow_fragments=False)
        bucket = parsed_url.netloc
        key = parsed_url.path.strip("/")

        s3_client = boto3.client("s3")

        try:
            response = s3_client.upload_file(local_path, bucket, key)
            signed_url = create_presigned_url(s3_path)

            st.sidebar.info(textwrap.dedent(f"""
            The report can be downloaded [here]({signed_url}).
            """))
        except ClientError as e:
            logging.error(e)

    return


def make_df_reports(bucket, prefix):
    s3 = boto3.client("s3")
    df = pd.DataFrame()
    df["filename"] = \
        [e['Key'] for p in s3.get_paginator("list_objects_v2")
            .paginate(Bucket=bucket, Prefix=prefix) for e in p['Contents']]
    #df["s3_path"] = "s3://" + bucket + "/" + df["filename"]
    df["filename"] = df["filename"].apply(os.path.basename)
    return df


#
# Panels
#
def make_mask(df, channel, family, item_id):
    mask = np.ones(len(df)).astype(bool)

    # only mask when all three keys are non-empty
    if channel == "" or family == "" or item_id == "":
        return ~mask

    mask &= df["channel"].str.upper() == channel.upper()
    mask &= df["family"].str.upper() == family.upper()
    mask &= df["item_id"].str.upper() == item_id.upper()

    return mask


@st.cache
def make_downloads(df_pred, df_results):
    """
    """

    pred_fn = os.path.join(ST_DOWNLOADS_PATH,
            f"{state.uploaded_file.name}_fcast.csv.gz")
    results_fn = os.path.join(ST_DOWNLOADS_PATH,
            f"{state.uploaded_file.name}_results.csv.gz")

    state.df_pred.to_csv(pred_fn, index=False, compression="gzip")
    state.df_results.to_csv(results_fn, index=False, compression="gzip")

    return pred_fn, results_fn


def panel_load_data():
    """Display the 'Load Data' panel.

    """

    def _load_data(path):
        if path.endswith(".csv"):
            compression = None
        elif path.endswith(".csv.gz"):
            compression = "gzip"
        else:
            raise NotImplementedError

        df = pd.read_csv(path,
            dtype={"timestamp": str, "channel": str, "family": str,
                   "item_id": str}, compression=compression)

        return df

    with st.beta_expander("☑️ Validate Data", expanded=True):
        _cols = st.beta_columns([3,1])

        with _cols[0]:
            fn = file_selectbox("File", args.local_dir)
            btn_refresh_files = st.button("Refresh Files")

        with _cols[1]:
            freq = st.selectbox("Frequency", list(FREQ_MAP.values()),
                    format_func=lambda s: FREQ_MAP_LONG[s])
            btn_validate = st.button("Validate")

        if fn is None:
            st.warning(textwrap.dedent("""
            **Warning**

            No files were detected.

            1. Upload your file(s).
            2. Click the **Refresh files** button.
            ⠀
            #### 
            """))

        if btn_validate:
            start = time.time()

            if fn is None:
                st.error(textwrap.dedent("""
                **Error**

                No files were selected.

                1. Upload your file(s).
                2. Click the **Refresh Files** button.
                3. Select the file from the dropdown box.
                4. Select the **Frequency**.
                5. Click the **Validate** button.

                ####
                """))
                st.stop()

            # temporarily load the file for validation and store it in state
            # iff the data is valid
            with st.spinner("Validating file ..."):
                #df, msgs, is_valid_file = validate(_load_data(fn))
                df, msgs, is_valid_file = validate(_load_data(fn))#.drop(["timestamp", "channel"], axis=1))

            if is_valid_file:
                with st.spinner("Processing file ..."):
                    state.report["data"]["path"] = fn
                    state.report["data"]["sz_bytes"] = os.path.getsize(fn)
                    state.report["data"]["freq"] = freq
                    state.report["data"]["df"] = \
                        load_data(df, impute_freq=state.report["data"]["freq"])

                    # clear any existing data health check results, this forces
                    # a rechecking of data health
                    state.report["data"]["df_health"] = None

                    st.text(f"(completed in {format_timespan(time.time() - start)})")
            else:
                err_bullets = "\n".join("- " + s for s in msgs["errors"])
                st.error(f"**Validation failed**\n\n{err_bullets}")


    return


def panel_load_report():
    """
    """

    def format_func(s):
        if s == "local":
            return "Local Filesystem"
        elif s == "s3":
            return "☁️ S3"

    s3 = boto3.client("s3")

    with st.beta_expander("⬆️ Load Report", expanded=True):
        report_source = st.radio("Source", ["local"], format_func=format_func)

        _cols = st.beta_columns([3,1])

        with _cols[0]:
            if report_source == "local":
                fn = file_selectbox("File", os.path.join(args.local_dir),
                                    globs=("*.pkl.gz",)) 
            elif report_source == "s3":
                # list the reports in the s3 bucket
    #           df_reports = make_df_reports(bucket, prefix)
    #           grid_resp = \
    #               display_ag_grid(df_reports, paginate=True, comma_cols=[],
    #                               selection_mode="single", use_checkbox=True)
    #           st.write(grid_resp)
                pass
            else:
                raise NotImplementedError
            load_report_btn = st.button("Load", key="load_report_btn")

        with _cols[1]:
            st.write("##")
            st.button("🔄", key="refresh_report_files_btn")

        if load_report_btn:
            with st.spinner("Loading Report ..."):
                state["report"] = cloudpickle.load(gzip.open(fn, "rb"))
    return


def panel_data_health():
    """
    """

    df = state.report["data"].get("df", None)
    df_health = state.report["data"].get("df_health", None)
    freq = state.report["data"].get("freq", None)

    if df is None:
        return

    st.header("Data Health")

    with st.beta_expander("❤️ Data Health", expanded=True):
        with st.spinner("Performing data health check ..."):
            start = time.time()

            # check iff required
            if df_health is None:
                df_health = make_health_summary(df, state.report["data"]["freq"])

                # save the health check results
                state.report["data"]["df_health"] = df_health

                # calc. ranked series by demand
                state.report["data"]["df_ranks"] = \
                    df.groupby(["channel", "family", "item_id"]) \
                      .agg({"demand": sum}) \
                      .sort_values(by="demand", ascending=False)

        num_series = df_health.shape[0]
        num_channels = df_health["channel"].nunique()
        num_families = df_health["family"].nunique()
        num_item_ids = df_health["item_id"].nunique()
        first_date = df_health['timestamp_min'].dt.strftime('%Y-%m-%d').min()
        last_date = df_health['timestamp_max'].dt.strftime('%Y-%m-%d').max()

        if freq == 'D':
            duration_unit = 'D'
            duration_str = 'days'
        elif freq in ("W", "W-MON",):
            duration_unit = 'W'
            duration_str = 'weeks'
        elif freq in ("M", "MS",):
            duration_unit = 'M'
            duration_str = 'months'
        else:
            raise NotImplementedError

        duration = pd.Timestamp(last_date).to_period(duration_unit) - \
                   pd.Timestamp(first_date).to_period(duration_unit)

        pc_missing = \
            df_health["demand_missing_dates"].sum() / df_health["demand_len"].sum()

        with st.beta_container():
            _cols = st.beta_columns(3)

            with _cols[0]:
                st.markdown("#### Summary")
                st.text(textwrap.dedent(f"""
                No. series:\t{num_series}  
                No. channels:\t{num_channels}  
                No. families:\t{num_families}  
                No. item IDs:\t{num_item_ids}
                """))

            with _cols[1]:
                st.markdown("#### Timespan")
                st.text(f"Frequency:\t{FREQ_MAP_LONG[freq]}\n"
                        f"Duration:\t{duration.n} {duration_str}\n"
                        f"First date:\t{first_date}\n"
                        f"Last date:\t{last_date}\n")
                        #f"% missing:\t{int(np.round(pc_missing*100,0))}")

            with _cols[2]:
                st.markdown("#### Timeseries Lengths")

                fig = pex.box(df_health, x="demand_nonnull_count", height=160)
                fig.update_layout(
                    margin={"t": 5, "b": 0, "r": 0, "l": 0},
                    xaxis_title=duration_str,
                    height=100
                )

                st.plotly_chart(fig, use_container_width=True)

            st.text(f"(completed in {format_timespan(time.time() - start)})")

    return


def panel_launch():
    """
    """

    def _format_func(short):
        if short == "local":
            s = " Local"
        if short == "lambdamap":
            s = "AWS Lambda"
        return s

    df = state.report["data"].get("df", None)
    df_health = state.report["data"].get("df_health", None)
    horiz = state.report["sfs"].get("horiz", None)
    freq = state.report["sfs"].get("freq", None)

    if df is None or df_health is None:
        return

    st.header("Statistical Forecasting")

    with st.beta_expander("🚀 Launch", expanded=True):
        st.write("")
        with st.form("sfs_form"):
            with st.beta_container():
                _cols = st.beta_columns(3)

                with _cols[0]:
                    horiz = st.number_input("Horizon Length", value=1, min_value=1)

                with _cols[1]:
                    freq = st.selectbox("Forecast Frequency", valid_launch_freqs(), 0,
                            format_func=lambda s: FREQ_MAP_LONG[s])

                with _cols[2]:
                    backend = st.selectbox("Compute Backend", ["lambdamap", "local"], 0, _format_func)

                btn_launch = st.form_submit_button("Launch")

        if btn_launch:
            start = time.time()

            # save form data
            state.report["sfs"]["freq"] = freq
            state.report["sfs"]["horiz"] = horiz
            state.report["sfs"]["backend"] = backend

            df = state.report["data"]["df"]
            freq_in = state.report["data"]["freq"]
            freq_out = state.report["sfs"]["freq"]

#           if backend == "lambdamap":
#               emoji = "λ"
#           elif backend == "local":
#               emoji = ":computer:"
#           else:
#               raise NotImplementedError

            with st.spinner(f":rocket: Launching forecasts ..."):
                if backend == "local":
                    wait_for = \
                        run_pipeline(df, freq_in, freq_out, obj_metric="smape_mean",
                            cv_stride=2, backend="futures", horiz=horiz)
                elif backend == "lambdamap":
                    wait_for = run_lambdamap(df, horiz, freq)
                else:
                    raise NotImplementedError

            display_progress(wait_for, "🔥 Generating forecasts")

            with st.spinner("Processing results ..."):
                raw_results = [f.result() for f in futures.as_completed(wait_for)]

                # generate the results and predictions as dataframes
                df_results, df_preds = process_forecasts(wait_for)
                df_results = df_results[df_results["rank"] == 1]

                # generate the demand classifcation info
                df_demand_cln = make_demand_classification(df, freq_in)

            # save results and forecast data
            state.report["sfs"]["df_results"] = df_results
            state.report["sfs"]["df_preds"] = df_preds
            state.report["sfs"]["df_demand_cln"] = df_demand_cln
            state.report["sfs"]["job_duration"] = time.time() - start

        job_duration = state.report["sfs"].get("job_duration", None)

        if job_duration:
            st.text(f"(completed in {format_timespan(job_duration)})")

    return


def panel_accuracy():
    """
    """

    df = state.report["data"].get("df", None)
    df_demand_cln = state.report["sfs"].get("df_demand_cln", None)
    df_results = state.report["sfs"].get("df_results", None)
    horiz = state.report["sfs"].get("horiz", None)
    freq_out = state.report["sfs"].get("freq", None)

    if df is None or df_results is None:
        return

    with st.beta_expander("🎯 Forecast Summary", expanded=True):
        df_cln = pd.DataFrame({"category": ["short", "medium", "continuous"]})
        df_cln = df_cln.merge(
            df_demand_cln["category"]
                .value_counts(normalize=True)
                .reset_index()
                .rename({"index": "category", "category": "frac"}, axis=1),
            on="category", how="left"
        )

        df_cln = df_cln.fillna(0.0)
        df_cln["frac"] *= 100
        df_cln["frac"] = df_cln["frac"].astype(int)

        _cols = st.beta_columns(3)

        with _cols[0]:
            st.markdown("#### Parameters")
            st.text(f"Horiz. Length:\t{horiz}\n"
                    f"Frequency:\t{FREQ_MAP_LONG[freq_out]}")

            st.markdown("#### Classification")
            st.text(f"Short:\t\t{df_cln.iloc[0]['frac']} %\n"
                    f"Medium:\t\t{df_cln.iloc[1]['frac']} %\n"
                    f"Continuous:\t{df_cln.iloc[2]['frac']} %")


        if "df_model_dist" not in state["report"]["sfs"]:
            # generate the performance summary (runtime)
            df_model_dist, best_err, naive_err = make_perf_summary(df_results)

            state["report"]["df_model_dist"] = df_model_dist
            state["report"]["best_err"] = best_err
            state["report"]["naive_err"] = naive_err
        else:
            df_model_dist = state["report"]["df_model_dist"]
            best_err = state["report"]["best_err"]
            naive_err = state["report"]["naive_err"]

        with _cols[1]:
            st.markdown("#### Best Models")
            df_model_dist = df_model_dist.query("perc > 0")
            labels = df_model_dist["model_type"].values
            values = df_model_dist["perc"].values

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.40)])
            fig.update(layout_showlegend=False)
            fig.update_layout(
                margin={"t": 0, "b": 0, "r": 20, "l": 20},
                width=200,
                height=150,
            )
            #fig.update_traces(textinfo="percent+label", texttemplate="%{label} – %{percent:.1%f}")
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig)

        acc = (1 - best_err.err_mean) * 100.
        acc_naive = (1 - naive_err.err_mean) * 100.

        with _cols[2]:
            st.markdown("#### Overall Accuracy")
            st.markdown(
                f"<div style='font-size:36pt;font-weight:bold'>{acc:.0f}%</div>"
                f"({np.clip(acc - acc_naive, 0, None):.0f}% increase vs. naive)", unsafe_allow_html=True)

    return


@st.cache()
def make_df_top(df, df_results, groupby_cols, dt_start, dt_stop, cperc_thresh):
    """
    """

    def calc_period_metrics(dd, dt_start, dt_stop):
        """
        """
        dt_start = pd.Timestamp(dt_start)
        dt_stop = pd.Timestamp(dt_stop)
        ts = np.hstack(dd["ts_cv"].apply(np.hstack))
        ix = (ts >= dt_start) & (ts <= dt_stop)
            
        ys = np.hstack(dd["y_cv"].apply(np.hstack))[ix]
        yp = np.hstack(dd["yp_cv"].apply(np.hstack))[ix]
        
        smape = calc_smape(ys, yp)
            
        return smape

    metric = "smape_mean"
    df.index.name = "timestamp"

    dt_start = pd.Timestamp(dt_start).strftime("%Y-%m-%d")
    dt_stop = pd.Timestamp(dt_stop).strftime("%Y-%m-%d")
    df2 = df.query(f"timestamp >= '{dt_start}' and timestamp <= '{dt_stop}'")
    total_demand = df2["demand"].sum()

    # calculate per-group demand %
    df_grp_demand = \
        df2.groupby(groupby_cols, as_index=False, sort=False) \
           .agg({"demand": sum})
    df_grp_demand["perc"] = df_grp_demand["demand"] / total_demand * 100

    # get the best models for each group
    df_grp_metrics = \
        df_results.query("rank == 1") \
            .groupby(groupby_cols, as_index=False, sort=False) \
            .apply(lambda dd: calc_period_metrics(dd, dt_start, dt_stop)) \
            .pipe(pd.DataFrame) \
            .rename({None: "smape"}, axis=1) \
            .reset_index()
    df_grp_metrics["accuracy"] = 100 * (1-df_grp_metrics["smape"])
    df_grp_metrics.drop(["index", "smape"], axis=1, inplace=True)

    # combine, sort, and display
    df_grp = df_grp_demand \
        .merge(df_grp_metrics, on=groupby_cols, how="left") \
        .sort_values(by="demand", ascending=False)
    df_grp["cperc"] = df_grp["perc"].cumsum()
    df_grp = df_grp.query(f"cperc <= {cperc_thresh}")
    df_grp.rename({"perc": "% total demand", "accuracy": "% accuracy"}, axis=1, inplace=True)
    df_grp.drop("cperc", axis=1, inplace=True)

    # calc. summary row
    df_grp_summary = df_grp.agg({"demand": sum, "% accuracy": "mean"})

    df_grp_summary["% total demand"] = np.round(100 * df_grp_summary["demand"] / total_demand, 1)
    df_grp_summary = pd.DataFrame(df_grp_summary).T[["demand", "% total demand", "% accuracy"]]
    df_grp_summary.insert(0, "group by", ", ".join(groupby_cols))
    df_grp_summary["% accuracy"] = df_grp_summary["% accuracy"].round(0)

    df_grp["demand"] = df_grp["demand"].round(0)
    df_grp["% total demand"] = df_grp["% total demand"].round(1)
    df_grp["% accuracy"] = df_grp["% accuracy"].round(0)
    df_grp.insert(0, "rank", np.arange(df_grp.shape[0]) + 1)

    df_grp_summary["demand"] = df_grp_summary["demand"].round(0)
    df_grp_summary["% total demand"] = df_grp_summary["% total demand"].round(1)
    
    return df_grp, df_grp_summary


def panel_top_performers():
    """
    """

    df = state.report["data"].get("df", None)
    df_demand_cln = state.report["sfs"].get("df_demand_cln", None)
    df_results = state.report["sfs"].get("df_results", None)
    horiz = state.report["sfs"].get("horiz", None)
    freq_out = state.report["sfs"].get("freq", None)

    if df is None or df_results is None:
        return

    with st.beta_expander("🏆 Top Performers", expanded=True):
        st.write("#### Filters")

        _cols = st.beta_columns([2,1,1])

        dt_min = df.index.min()
        dt_max = df.index.max()

        with _cols[0]:
            groupby_cols = st.multiselect("Group By",
                ["channel", "family", "item_id"], ["channel", "family", "item_id"])

        with _cols[1]:
            dt_start = st.date_input("Start", value=dt_min, min_value=dt_min, max_value=dt_max)
        with _cols[2]:
            dt_stop = st.date_input("Stop", value=dt_max, min_value=dt_min, max_value=dt_max)

        cperc_thresh = st.slider("Percentage of total demand",
            step=5, value=80, format="%d%%")

        dt_start = dt_start.strftime("%Y-%m-%d")
        dt_stop = dt_stop.strftime("%Y-%m-%d")

        start = time.time()

        with st.spinner("Processing top performers ..."):
            df_grp, df_grp_summary = \
                make_df_top(df, df_results, groupby_cols, dt_start, dt_stop, cperc_thresh)

        st.write("#### Group Summary")

        with st.spinner("Loading **Summary** table"):
            display_ag_grid(df_grp_summary, auto_height=True,
                comma_cols=("demand",))

        st.write("#### Groups")
        with st.spinner("Loading **Groups** table ..."):
            display_ag_grid(df_grp, paginate=True, comma_cols=("demand",))

        st.text(f"(completed in {format_timespan(time.time() - start)})")

        if st.button("Export"):
            with st.spinner("Exporting **Top Performers** ..."):
                start = time.time()

                # write the dataframe to s3
                now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                basename = os.path.basename(state["report"]["data"]["path"])
                s3_sfs_export_path = state["report"]["sfs"]["s3_sfs_export_path"]
                s3_path = f'{s3_sfs_export_path}/{basename}_{now_str}_sfs-top-performers.csv.gz'

                wr.s3.to_csv(df_grp, s3_path, compression="gzip", index=False)

                # generate presigned s3 url for user to download
                signed_url = create_presigned_url(s3_path)

                st.info(textwrap.dedent(f"""
                Download the top performers file [here]({signed_url})  
                `(completed in {format_timespan(time.time() - start)})`
                """))

    return


def panel_visualization():
    """
    """

    df = state.report["data"].get("df", None)
    df_results = state.report["sfs"].get("df_results", None)
    df_preds = state.report["sfs"].get("df_preds", None)

    if df is None or df_results is None or df_preds is None:
        return

    freq = state.report["sfs"]["freq"]
    horiz = state.report["sfs"]["horiz"]
    start = time.time()

    df_top = df.groupby(["channel", "family", "item_id"], as_index=False) \
               .agg({"demand": sum}) \
               .sort_values(by="demand", ascending=False)

    channel_vals = [""] + sorted(df_results["channel"].unique())
    family_vals = [""] + sorted(df_results["family"].unique())
    item_id_vals = [""] + sorted(df_results["item_id"].unique())

    channel_index = channel_vals.index(df_top["channel"].iloc[0])
    family_index = family_vals.index(df_top["family"].iloc[0])
    item_id_index = item_id_vals.index(df_top["item_id"].iloc[0])

    with st.beta_expander("👁️  Visualization", expanded=True):
        with st.form("viz_form"):
            st.markdown("#### Filter By")
            _cols = st.beta_columns(3)

            with _cols[0]:
                channel_choice = st.selectbox("Channel", channel_vals, index=channel_index)
            with _cols[1]:
                family_choice = st.selectbox("Family", family_vals, index=family_index)
            with _cols[2]:
                item_id_choice = st.selectbox("Item ID", item_id_vals, index=item_id_index)

            viz_form_button = st.form_submit_button("Apply")

        if viz_form_button:
            pass

        results_mask = \
            make_mask(df_results, channel_choice, family_choice, item_id_choice)
        pred_mask = \
            make_mask(df_preds, channel_choice, family_choice, item_id_choice)

        df_plot = df_preds[pred_mask]

        if len(df_plot) > 0:
            # display the line chart

            y = df_plot.query("type == 'actual'")["demand"]
            y_ts = df_plot.query("type == 'actual'")["timestamp"]

            yp = df_plot.query("type == 'fcast'")["demand"]
            yp_ts = df_plot.query("type == 'fcast'")["timestamp"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_ts, y=y, mode='lines', name="actual",
                fill="tozeroy", line={"width": 3.5}, marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=yp_ts, y=yp, mode='lines', name="forecast",
                fill="tozeroy", line={"width": 3.5}, marker=dict(size=4)
            ))

            # plot 
            dd = df_results[results_mask].query("rank == 1").iloc[0]

            df_backtest = \
                pd.DataFrame({"yp": np.hstack(dd['yp_cv'])},
                             index=pd.DatetimeIndex(np.hstack(dd["ts_cv"]))) \
                  .sort_index() \
                  .resample(FREQ_MAP_PD[freq]) \
                  .apply(np.nanmean)

            fig.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest.yp, mode="lines",
                name="backtest", line_dash="dot", line_color="black"))
                

    #       fig.update_layout(
    #           xaxis={
    #               "showgrid": True,
    #               "gridcolor": "lightgrey",
    #           },
    #           yaxis={
    #               "showgrid": True,
    #               "gridcolor": "lightgrey",
    #           }
    #       )
            fig.update_layout(
                margin={"t": 0, "b": 0, "r": 0, "l": 0},
                height=250,
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor":"left", "x": 0.0}
            )
            fig.update_xaxes(
                rangeslider_visible=True,
#               rangeselector=dict(
#                   buttons=list([
#                       dict(count=1, label="1m", step="month", stepmode="backward"),
#                       dict(count=6, label="6m", step="month", stepmode="backward"),
#                       dict(count=1, label="YTD", step="year", stepmode="todate"),
#                       dict(count=1, label="1y", step="year", stepmode="backward"),
#                       dict(step="all")
#                   ])
#               )
            )

            initial_range = pd.date_range(end=yp_ts.max(), periods=horiz*8, freq=freq)
            initial_range = [max(initial_range[0], y_ts.min()), initial_range[-1]]

            fig["layout"]["xaxis"].update(range=initial_range)

            st.plotly_chart(fig, use_container_width=True)

        plot_duration = time.time() - start
        st.text(f"(completed in {format_timespan(plot_duration)})")

    return


def panel_downloads():
    """
    """

    df = state.report["data"].get("df", None)
    df_results = state.report["sfs"].get("df_results", None)
    df_preds = state.report["sfs"].get("df_preds", None)

    if df is None or df_results is None or df_preds is None:
        return

    with st.beta_expander("⬇️  Downloads", expanded=True):
        export_forecasts_btn = \
            st.button("Export Forecasts", key="sfs_export_forecast_btn")

        if export_forecasts_btn:
            start = time.time()
            s3_sfs_export_path = state["report"]["sfs"]["s3_sfs_export_path"]

            with st.spinner("Exporting Forecasts ..."):
                now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                basename = os.path.basename(state["report"]["data"]["path"])
                s3_forecasts_path = f'{s3_sfs_export_path}/{basename}_{now_str}_sfs-forecast.csv.gz'
                wr.s3.to_csv(df_preds, s3_forecasts_path, compression="gzip", index=False)
                forecasts_signed_url = create_presigned_url(s3_forecasts_path)

            st.info(textwrap.dedent(f"""
            Download the forecasts file [here]({forecasts_signed_url})  
            `(completed in {format_timespan(time.time()-start)})`.  
            """))

            with st.spinner("Exporting Backtests ..."):
                #backtests_path = os.path.join(ST_STATIC_PATH, f"{basename}-sfs-backtests.csv.gz")
                df_backtests = make_df_backtests(df_results)
                s3_backtests_path = f'{s3_sfs_export_path}/{basename}_{now_str}_sfs-backtests.csv.gz'
                wr.s3.to_csv(df_backtests, s3_backtests_path, compression="gzip", index=False)
                backtests_signed_url = create_presigned_url(s3_backtests_path)

            st.info(textwrap.dedent(f"""
            Download the backtests file [here]({backtests_signed_url})  
            `(completed in {format_timespan(time.time()-start)})`.
            """))

#               df_backtests.to_csv(backtests_path, compression="gzip", index=False)
#                   st.info(textwrap.dedent(f"""
#                   Download the forecasts file [here]({forecasts_signed_url}).  

#           st.write(forecasts_path)
#           st.write(backtests_path)
#               if False:
#                   now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#                   basename = os.path.basename(state["report"]["data"]["path"])
#                   s3_sfs_export_path = state["report"]["sfs"]["s3_sfs_export_path"]
#                   s3_forecasts_path = f'{s3_sfs_export_path}/{basename}_{now_str}_sfs-forecast.csv.gz'

#                   wr.s3.to_csv(df_preds, s3_forecasts_path, compression="gzip", index=False)

#                   forecasts_signed_url = create_presigned_url(s3_forecasts_path)

#                   s3_backtests_path = f'{s3_sfs_export_path}/{basename}_{now_str}_sfs-backtests.csv.gz'

#                   df_backtests = make_df_backtests(df_results)
#                   wr.s3.to_csv(df_backtests, s3_backtests_path, compression="gzip", index=False)
#                   backtests_signed_url = create_presigned_url(s3_backtests_path)


#                   plot_duration = time.time() - start
#                   st.text(f"(completed in {format_timespan(plot_duration)})")

    return


#
# ML Forecasting Panels
#
def parse_s3_json(path):
    """
    """

    parsed_url = urlparse(path, allow_fragments=False)
    bucket = parsed_url.netloc
    key = parsed_url.path.strip("/")

    s3 = boto3.resource("s3")
    s3_obj = s3.Object(bucket_name=bucket, key=key)

    status_dict = json.loads(s3_obj.get()["Body"].read())

    return status_dict


def panel_ml_launch():
    """
    """
    df = state.report["data"].get("df", None)

    if df is None:
        return

    st.header(":construction: ML Forecasting :construction:")

    with st.beta_expander("🚀 Launch"):
        with st.form("ml_form"):
            _cols = st.beta_columns(3)

            with _cols[0]:
                horiz = st.number_input("Horizon Length", key="ml_horiz_input",
                                        value=1, min_value=1)

            with _cols[1]:
                freq = st.selectbox("Forecast Frequency",
                    valid_launch_freqs(), 0,
                    format_func=lambda s: FREQ_MAP_LONG[s], key="ml_freq_input")

            with _cols[2]:
                st.selectbox("Algorithm", ["AutoML"], 0, key="ml_algo_input")

            ml_form_button = st.form_submit_button("Launch")

        # Launch Amazon Forecast job
        if ml_form_button:
            with st.spinner("Launching ML forecasting job ..."):
                state.report["afc"]["horiz"] = horiz
                state.report["afc"]["freq"] = freq

                execution_arn, prefix, status_json_s3_path = \
                    run_ml_state_machine()

                state.report["afc"]["execution_arn"] = execution_arn
                state.report["afc"]["status_json_s3_path"] = status_json_s3_path
                state.report["afc"]["prefix"] = prefix

        execution_arn = state.report["afc"].get("execution_arn", None)
        ml_refresh_job_button = st.button("Refresh Job Status")

        if ml_refresh_job_button:
            if execution_arn is None:
                st.warning("Job not yet launched")
                return

            with st.spinner("Checking job status ..."):
                sfn_status, status_dict = refresh_ml_state_machine_status()
                sfn_state = status_dict["PROGRESS"]["state"]

                if sfn_status not in ("RUNNING", "SUCCEEDED", "FAILED",
                                      "TIMED_OUT", "ABORTED",):
                    sfn_status_str = "UNKNOWN"

                st.info(textwrap.dedent(f"""
                **Status:** {sfn_status}  
                **Stage:** {sfn_state}  
                **AWS Console:** [view](https://console.aws.amazon.com/states/home#/executions/details/{execution_arn})
                """))

                ml_stop_button = st.button("Stop Job")

                if state.report["afc"].get("status_json_s3_path", None):
                    status_dict = parse_s3_json(state.report["afc"]["status_json_s3_path"])
                    prefix = status_dict["prefix"]
                    s3_export_path = status_dict["s3_export_path"]

                    preds_s3_prefix = \
                        f'{s3_export_path}/{prefix}/{prefix}_processed.csv'
                    results_s3_prefix = \
                        f'{s3_export_path}/{prefix}/accuracy-metrics-values/Accuracy_{prefix}_*.csv'

                    try:
                        df_results = wr.s3.read_csv(results_s3_prefix,
                            dtype={"channel": str, "family": str, "item_id": str})
                        df_results[["channel", "family", "item_id"]] = \
                            df_results["item_id"].str.split("@@", expand=True)
                        state.report["afc"]["df_results"] = df_results
                        state.report["afc"]["results_s3_prefix"] = results_s3_prefix
                    except NoFilesFound:
                        st.warning("The forecast results file is not yet ready.")

                    try:
                        df_preds = wr.s3.read_csv(preds_s3_prefix,
                            dtype={"channel": str, "family": str, "item_id": str})
                        df_preds["type"] = "fcast"

                        df_preds = df_preds.append(
                            load_data(df, FREQ_MAP_PD[state.report["afc"]["freq"]])
                                .reset_index()
                                .rename({"index": "timestamp"}, axis=1)
                                .assign(type='actual'))

                        state.report["afc"]["df_preds"] = df_preds
                        state.report["afc"]["preds_s3_prefix"] = preds_s3_prefix
                    except NoFilesFound:
                        st.warning("The forecast predictions file is not yet ready.")

            if ml_stop_button:
                sfn_client = boto3.client("stepfunctions")
                resp = sfn_client.stop_execution(executionArn=execution_arn)
                st.write(resp)

    return


def panel_ml_forecast_summary():
    """
    """

    df = state.report["data"].get("df", None)
    df_results = state.report["afc"].get("df_results", None)

    if df is None or df_results is None:
        return

    with st.beta_expander("Forecast Summary", expanded=False):
        ml_acc = 100 - np.nanmean(df_results.query("backtest_window == 'Summary'")["WAPE"].clip(0,100))

        _cols = st.beta_columns(3)

        with _cols[0]:
            st.markdown("#### Overall Accuracy")
            st.markdown(
                f"<div style='font-size:36pt;font-weight:bold'>{ml_acc:.0f}%</div>",
                unsafe_allow_html=True)

        with _cols[1]:
            pass

        with _cols[2]:
            pass

    return


def panel_ml_visualization():
    """
    """

    df = state.report["data"].get("df", None)
    df_ml_results = state.report["afc"].get("df_results", None)
    df_ml_preds = state.report["afc"].get("df_preds", None)

    if df is None or df_ml_results is None or df_ml_preds is None:
        return

    freq = state.report["afc"]["freq"]
    horiz = state.report["afc"]["horiz"]
    start = time.time()

    df_top = df.groupby(["channel", "family", "item_id"], as_index=False) \
               .agg({"demand": sum}) \
               .sort_values(by="demand", ascending=False)

    channel_vals = [""] + sorted(df_ml_results["channel"].unique())
    family_vals = [""] + sorted(df_ml_results["family"].unique())
    item_id_vals = [""] + sorted(df_ml_results["item_id"].unique())

    channel_index = channel_vals.index(df_top["channel"].iloc[0])
    family_index = family_vals.index(df_top["family"].iloc[0])
    item_id_index = item_id_vals.index(df_top["item_id"].iloc[0])

    with st.beta_expander("👁️  Visualization", expanded=True):
        with st.form("ml_viz_form"):
            st.markdown("#### Filter By")
            _cols = st.beta_columns(3)

            with _cols[0]:
                channel_choice = st.selectbox("Channel", channel_vals, index=channel_index, key="ml_results_channel")

            with _cols[1]:
                family_choice = st.selectbox("Family", family_vals, index=family_index, key="ml_results_family")

            with _cols[2]:
                item_id_choice = st.selectbox("Item ID", item_id_vals, index=item_id_index, key="ml_results_item")

            viz_form_button = st.form_submit_button("Apply")

        if viz_form_button:
            pass

        results_mask = \
            make_mask(df_ml_results, channel_choice, family_choice, item_id_choice)
        pred_mask = \
            make_mask(df_ml_preds, channel_choice, family_choice, item_id_choice)

        df_plot = df_ml_preds[pred_mask]

        if len(df_plot) > 0:
            # display the line chart
            #fig = pex.line(df_plot, x="timestamp", y="demand", color="type")

            y = df_plot.query("type == 'actual'")["demand"]
            y_ts = df_plot.query("type == 'actual'")["timestamp"]

            yp = df_plot.query("type == 'fcast'")["demand"]
            yp_ts = df_plot.query("type == 'fcast'")["timestamp"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_ts, y=y, mode='lines+markers', name="actual",
                fill="tozeroy", line={"width":3}, marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=yp_ts, y=yp, mode='lines+markers', name="forecast",
                fill="tozeroy", marker=dict(size=4)
            ))
            fig.update_layout(
                margin={"t": 0, "b": 0, "r": 0, "l": 0},
                height=250,
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor":"left", "x": 0.0}
            )

            fig.update_xaxes(rangeslider_visible=True)

            initial_range = pd.date_range(end=yp_ts.max(), periods=horiz*8, freq=freq)
            initial_range = [max(initial_range[0], y_ts.min()), initial_range[-1]]

            fig["layout"]["xaxis"].update(range=initial_range)
            st.plotly_chart(fig, use_container_width=True)

        plot_duration = time.time() - start
        st.text(f"(completed in {format_timespan(plot_duration)})")

    return


def run_ml_state_machine():
    """Execute the Amazon Forecast state machine.

    """
    PD_TIMESTAMP_FMT = "%Y-%m-%d"
    AFC_TIMESTAMP_FMT = "yyyy-MM-dd"
    AFC_FORECAST_HORIZON = state.report["afc"]["horiz"]
    AFC_FORECAST_FREQUENCY = state.report["afc"]["freq"]

    df = state.report["data"].get("df", None)
    fn = state.report["data"]["path"]

    assert(df is not None)

    # state.df is already resampled to same frequency as the forecast freq.
    AFC_DATASET_FREQUENCY = AFC_FORECAST_FREQUENCY

    state_machine_arn = None

    # generate a unique prefix for the Amazon Forecast resources
    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = f"SfsAfc{now_str}"

    # get the state machine arn and s3 paths
    ssm_client = boto3.client("ssm")
    state_machine_arn = \
        ssm_client.get_parameter(Name="SfsAfcStateMachineArn")["Parameter"]["Value"]
    s3_input_path = \
        ssm_client.get_parameter(Name="SfsS3InputPath")["Parameter"]["Value"].rstrip("/")
    s3_output_path = \
        ssm_client.get_parameter(Name="SfsS3OutputPath")["Parameter"]["Value"].rstrip("/")

    # generate amazon forecast compatible data
    with st.spinner("Launching Amazon Forecast job ..."):
        df_afc = df \
            | px.reset_index() \
            | px.rename({"index": "timestamp"}, axis=1) \
            | px.assign(item_id=px["channel"] + "@@" + px["family"] + "@@" + px["item_id"]) \
            | px[["timestamp", "demand", "item_id"]] \
            | px.sort_values(by=["item_id", "timestamp"])

        df_afc["timestamp"] = \
            pd.DatetimeIndex(df_afc["timestamp"]).strftime("%Y-%m-%d") 
        afc_input_fn = \
            re.sub("(.csv.gz)", ".csv", os.path.basename(fn))
        s3_input_path = f"{s3_input_path}/{afc_input_fn}"

        # upload the input csv to s3
        wr.s3.to_csv(df_afc, s3_input_path, index=False)

        # upload local re-sampled csv file to s3 input path
        client = boto3.client("stepfunctions")

        resp = client.start_execution(
            stateMachineArn=state_machine_arn,
            input=json.dumps({
                "prefix": prefix,
                "data_freq": AFC_DATASET_FREQUENCY,
                "horiz": AFC_FORECAST_HORIZON,
                "freq": AFC_FORECAST_FREQUENCY,
                "s3_path": s3_input_path,
                "s3_export_path": s3_output_path
            })
        )

    status_json_s3_path = \
        os.path.join(s3_output_path, f'{prefix}_status.json')

    return resp["executionArn"], prefix, status_json_s3_path


def refresh_ml_state_machine_status():
    """
    """
    sfn_client = boto3.client("stepfunctions")
    resp = sfn_client.describe_execution(
        executionArn=state.report["afc"]["execution_arn"])
    sfn_status = resp["status"]
    status_dict = parse_s3_json(state.report["afc"]["status_json_s3_path"])
    return sfn_status, status_dict


def file_selectbox(label, folder, globs=("*.csv", "*.csv.gz")):
    """
    """

    if folder.startswith("s3://"):
        raise NotImplementedError
    else:
        fns = []
        for pat in globs:
            fns.extend(glob.glob(os.path.join(folder, pat)))

    fn = st.selectbox(label, fns, format_func=lambda s: os.path.basename(s))

    return fn


def nav_radio_format_func(s):
    if s == "create_report":
        return "📄 Create Report"
    elif s == "load_report":
        return "⬆️ Load Report"
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--local-dir", type=str,
        help="/path/to local folder to store input/output files.",
        default=os.path.expanduser("~/SageMaker/"))

    parser.add_argument("--lambdamap-function", type=str,
        help="ARN/name of the lambdamap function",
        default="SfsLambdaMapFunction")

    args = parser.parse_args()

    assert(os.path.exists(os.path.expanduser(args.local_dir)))

    #st.set_page_config(layout="wide")

    #
    # Sidebar
    #
    st.sidebar.title("Amazon Simple Forecast Accelerator")
    st.sidebar.markdown(textwrap.dedent("""
    - [github](https://github.com/aws-samples/simple-forecast-solution)
    """))

    nav_radio_btn = \
        st.sidebar.radio("Navigation", ["create_report", "load_report"],
                         format_func=nav_radio_format_func)

    save_report_btn = st.sidebar.button("💾 Save Report")
    clear_report_btn = st.sidebar.button("❌ Clear Report")

    if save_report_btn:
        save_report()

    if clear_report_btn:
        state.pop("report")

    if "report" not in state:
        state["report"] = {"data": {}, "sfs": {}, "afc": {}}

    # populate state global variables from ssm
    ssm_client = boto3.client("ssm")

    if "s3_afc_export_path" not in state["report"]["afc"]:
        state["report"]["afc"]["s3_afc_export_path"] = \
            ssm_client.get_parameter(Name="SfsS3OutputPath")["Parameter"]["Value"].rstrip("/")

    if "s3_bucket" not in state["report"]:
        state["report"]["s3_bucket"] = \
            ssm_client.get_parameter(Name="SfsS3Bucket")["Parameter"]["Value"].strip("/")

    if "s3_sfs_export_path" not in state["report"]:
        state["report"]["sfs"]["s3_sfs_export_path"] = \
            f's3://{state["report"]["s3_bucket"]}/sfs-exports'

    if "s3_sfs_reports_path" not in state["report"]:
        state["report"]["sfs"]["s3_sfs_reports_path"] = \
            f's3://{state["report"]["s3_bucket"]}/sfs-reports'

    #
    # Main page
    #
    st.subheader("Amazon Simple Forecast Accelerator")

    if nav_radio_btn == "create_report":
        st.title("Create Report")
        st.markdown("")
        panel_load_data()
    elif nav_radio_btn == "load_report":
        st.title("Load Report")
        st.markdown("")
        panel_load_report()

    panel_data_health()

    panel_launch()
    panel_accuracy()
    panel_top_performers()
    panel_visualization()
    panel_downloads()

    panel_ml_launch()
    panel_ml_forecast_summary()
    panel_ml_visualization()
