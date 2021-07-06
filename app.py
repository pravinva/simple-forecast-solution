import time
import datetime
import uuid
import base64

import numpy as np
import pandas as pd
import awswrangler as wr
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from collections import OrderedDict
from concurrent import futures
from tabulate import tabulate
from stqdm import stqdm
from SessionState import get_state
from sfs import (load_data, run_pipeline,
    make_demand_classification, make_perf_summary,
    make_health_summary, GROUP_COLS, EXP_COLS)

FREQ_MAP = OrderedDict(Daily="D", Weekly="W", Monthly="M")


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

    return msgs, is_valid_file

@st.cache
def load_uploaded_file(uploaded_file):
    """
    """

    if uploaded_file.name.endswith(".csv.gz"):
        df = pd.read_csv(uploaded_file, compression="gzip")
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raise NotImplementedError

    # reset read position to start of file
    uploaded_file.seek(0, 0)

    return df


#
# SFS interaction
#
def launch_sfs_forecast(state):
    """
    """

    assert state.df is not None

    freq = FREQ_MAP[state.out_freq]
    n_series = state.df.groupby(GROUP_COLS).ngroups

    # generate the pipeline
    gen_pipeline = run_pipeline(state.df, state.horiz, freq)

    # get the best model (and forecast) for each timeseries
    results_lst = []
    pred_lst = []

    for dd_results, dd_pred in stqdm(gen_pipeline, total=n_series, leave=True):
        results_lst.append(dd_results[dd_results["rank"] == 1])
        pred_lst.append(dd_pred)

    # save the results for the report view page
    state.df_results = pd.concat(results_lst) \
                         .drop("rank", axis=1) \
                         .reset_index(drop=True)
    state.df_pred = pd.concat(pred_lst) \
                      .reset_index(drop=True)

    return


#
# Panels
#
def panel_health_check(state):
    st.header("Data Health Check")

    with st.spinner("Running data health check..."):
        df_health = make_health_summary(state.df, FREQ_MAP[state.freq_in])
        state.df_health = df_health

    _cols = st.beta_columns([1,1,1])

    with _cols[0]:
        st.subheader("Summary")

        num_series = df_health.shape[0]
        num_channels = df_health["channel"].nunique()
        num_families = df_health["family"].nunique()
        num_item_ids = df_health["item_id"].nunique()
        first_date = df_health['timestamp_min'].dt.strftime('%Y-%m-%d').min()
        last_date = df_health['timestamp_max'].dt.strftime('%Y-%m-%d').max()

        state.num_series = num_series

        st.text(f"No. series:\t{num_series}\n"
                f"No. channels:\t{num_channels}\n"
                f"No. families:\t{num_families}\n"
                f"No. item IDs:\t{num_item_ids}\n"
                f"First date:\t{first_date}\n"
                f"Last date:\t{last_date}")

    with _cols[1]:
        st.subheader("Series Lengths")
        fig = px.histogram(df_health, x="demand_nonnull_count",
                           height=160)
        fig.update_layout(
            margin={"t": 0, "b": 5, "r": 5, "l": 5},
            xaxis_title="length",
            yaxis_title="# series",
            bargroupgap=0.025,
        )

        st.plotly_chart(fig, use_container_width=True)

    with _cols[2]:
        st.subheader("Missing Dates")
        fig = px.histogram(df_health, x="demand_missing_dates",
                           height=160)
        fig.update_layout(
            margin={"t": 0, "b": 5, "r": 5, "l": 5},
            xaxis_title="# dates",
            yaxis_title="# series",
            bargroupgap=0.025,
        )

        st.plotly_chart(fig, use_container_width=True)

    return


def panel_launch_forecast(state):
    """
    """

    with st.beta_container():
        st.header("Launch Forecast")

        with st.form("form_create_forecast"):
            _cols = st.beta_columns(2)

            with _cols[0]:
                horiz = st.number_input("Forecast horizon length", min_value=1)

            with _cols[1]:
                freq_out = st.selectbox("Forecast Frequency", list(FREQ_MAP.keys()))

            create_forecast_button = st.form_submit_button("Launch")

    if create_forecast_button:
        state.horiz = horiz
        state.freq_out = freq_out
        wait_for = \
            run_pipeline(state.df, state.horiz, FREQ_MAP[state.freq_in],
                FREQ_MAP[state.freq_out], obj_metric="smape_mean", cv_stride=4,
                backend="futures")

        display_progress(wait_for)

        # aggregate the forecasts
        raw_results = [f.result() for f in futures.as_completed(wait_for)]

        pred_lst = []
        results_lst = []

        for df_pred, df_results in raw_results:
            pred_lst.append(df_pred)
            results_lst.append(df_results)

        # results dataframe
        state.df_results = pd.concat(results_lst) \
                             .reset_index(drop=True)

        # predictions dataframe
        state.df_pred = pd.concat(pred_lst)
        state.df_pred.index.name = "timestamp"
        state.df_pred.reset_index(inplace=True)

        # analysis dataframes
        state.df_demand_cln = \
            make_demand_classification(state.df, FREQ_MAP[state.freq_out])
        state.perf_summary = make_perf_summary(state.df_results)

    return


#
# Pages
#
def page_upload_file(state):
    """
    """

    with st.beta_container():
        st.header("Select File")

        with st.form("form_select_file"): 
            state.uploaded_file = st.file_uploader("Select a .csv or .csv.gz file")

            _cols = st.beta_columns(1)

            with _cols[0]:
                state.freq_in = st.selectbox("Input Frequency", list(FREQ_MAP.keys())) 

            validate_button = st.form_submit_button("Validate")

        #
        # run validation process
        #
        if validate_button:
            #st.text(state.uploaded_file)
            if state.uploaded_file:
                with st.spinner("Validating file..."):
                    state.df_upload = load_uploaded_file(state.uploaded_file)
                    state.vldn_msgs, state.is_valid_file = validate(state.df_upload)
            else:
                st.info("Please select a file")

        # validation error messages
        if state.is_valid_file:
            # parse data into sfs format and impute missing dates according
            # to the input frequency
            state.df = load_data(state.df_upload, impute_freq=FREQ_MAP[state.freq_in])
        elif state.is_valid_file is None:
            pass
        else:
            err_bullets = \
                "\n\n".join("- " + s for s in state.vldn_msgs["errors"])
            st.error(f"Validation failed\n\n{err_bullets}")
            st.stop()

        #
        # run data health check
        #
        if state.is_valid_file and state.df is not None:
            panel_health_check(state)
            panel_launch_forecast(state)

    return


def page_create_forecast(state):
    """
    """

    freq_options = ["Daily", "Weekly", "Monthly"]

    with st.beta_container():
        st.header("Create Forecast")

        # Display input file info.
        if state.uploaded_file:
            file_col1, file_col2, file_col3 = st.beta_columns([1,1,1])

            with file_col1:
                st.markdown(f"**{state.uploaded_file.name}**")

            with file_col2:
                st.markdown(f"Size: ~{state.uploaded_file.size / 1000**2:.1f} MB")
        else:
            state.uploaded_file = st.file_uploader("Select File")

            if state.uploaded_file:
                df, msgs, is_valid_file = validate(state.uploaded_file)
                state.is_valid_file = is_valid_file
                state.df = df

        col1, col2, col3 = st.beta_columns(3)

        with col1:
            state.in_freq = \
                st.selectbox("Input Frequency",
                    freq_options,
                    freq_options.index(state.in_freq) if state.in_freq else 0
                )

        with col2:
            state.horiz = \
                st.number_input("Forecast Horizon Length",
                    value=(state.horiz or 1), min_value=1)

        with col3:
            state.out_freq = \
                st.selectbox("Forecast Frequency",
                    freq_options,
                    freq_options.index(state.out_freq) if state.out_freq else 0)

        launch_button = st.button(label="Launch")

        if launch_button and state.is_valid_file:
            launch_sfs_forecast(state)

    return


def page_view_report(state):
    """
    """

    def make_mask(df, channel, family, item_id):
        mask = np.ones(len(df)).astype(bool)

        # only mask when all three keys are non-empty
        if channel == "" or family == "" or item_id == "":
            return ~mask

        mask &= df["channel"] == channel
        mask &= df["family"] == family
        mask &= df["item_id"] == item_id

        return mask

    def _plot_demand_classification(df_demand_cln):
        df_plot = pd.DataFrame({"category": ["short", "medium", "continuous"]})
        df_plot = df_plot.merge(
            df_demand_cln["category"].value_counts(normalize=True) \
                                   .reset_index() \
                                   .rename({"index": "category",
                                            "category": "frac"}, axis=1),
            on="category", how="left"
        )
        df_plot["frac"] *= 100

        fig = go.Figure(
            go.Bar(
                x=df_plot["category"], y=df_plot["frac"],
            )
        )

        fig.update_layout(
            margin={"t": 20, "b": 0, "r": 20, "l": 20},
            width=250,
            height=260,
        )

        st.plotly_chart(fig)

        return
        
    st.header("Report")

    if state.df_results is None or state.df_pred is None:
        st.text("Results not ready")
        st.stop()

    with st.beta_container():
        df_pred = state.df_pred

        if df_pred is None:
            st.markdown("Results not yet ready")
            return

        df_hist = state.df_pred.query("type == 'actual'")
        df_results = state.df_results \
                          .assign(_index="") \
                          .set_index("_index")
        
        channel_vals = [""] + sorted(df_results["channel"].unique())
        family_vals = [""] + sorted(df_results["family"].unique())
        item_id_vals = [""] + sorted(df_results["item_id"].unique())

        num_series = df_hist[["channel", "family", "item_id"]] \
                        .drop_duplicates() \
                        .shape[0]

        df_model_dist, sr_err, sr_err_naive, acc_increase = \
            make_perf_summary(df_results)

        cols = st.beta_columns(4)

        with cols[0]:
            n_top = 10
            st.subheader(f"Top {n_top}")
            st.markdown("#### By Demand")

            df_top = df_hist.groupby(GROUP_COLS, as_index=False) \
                            .agg({"demand": sum}) \
                            .sort_values(by="demand", ascending=False) \
                            .head(n_top) \
                            .reset_index(drop=True)

            #df_top["demand"] = df_top["demand"].apply(lambda x: f"{x:,.0f}")
            df_top = df_top.assign(_index=np.arange(n_top)+1).set_index("_index")

            #st.text(df_top.to_markdown(index=False, tablefmt="simple", floatfmt=",.0f"), headers=[])
            st.text(tabulate(df_top, floatfmt=",.0f", showindex="never",
                tablefmt="plain", headers="keys"))

        with cols[1]:
            st.subheader("Summary")
            st.markdown("#### Historical")
            st.text(f"No. Series: {num_series}\n"
                    f"Frequency:  {state.freq_in}\n"
                    f"Channels:   {len(channel_vals)-1}\n"
                    f"Families:   {len(family_vals)-1}\n"
                    f"Items:      {len(item_id_vals)-1}\n"
            )

            st.markdown("#### Forecast")
            st.text(f"Horizon:    {state.horiz}\n"
                    f"Frequency:  {state.freq_out}")

        with cols[2]:
            st.subheader("Demand Classification")
            _plot_demand_classification(state.df_demand_cln)

        with cols[3]:
            acc = (1 - sr_err.err_mean) * 100.
            acc_naive = (1 - sr_err_naive.err_mean) * 100.

            st.subheader("Performance")
            st.markdown("#### Forecast Accuracy")
            st.markdown(f"## {acc:.0f}%")
            st.markdown(f"(_{acc - acc_naive:.0f}% increase vs. naive_)")
            st.markdown("#### Best performing models")

            df_model_dist = df_model_dist.query("perc > 0")
            labels = df_model_dist["model_type"].values
            values = df_model_dist["perc"].values

            print(df_model_dist)

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
            fig.update(layout_showlegend=False)
            fig.update_layout(
                margin={"t": 20, "b": 0, "r": 20, "l": 20},
                width=200,
                height=200,
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            st.plotly_chart(fig)


        st.subheader("Visualization")
        col1, col2 = st.beta_columns([1,4])

        with col1:
            st.markdown("#### Filter")

            # get default choices
            channel_index = channel_vals.index(df_top["channel"].iloc[0])
            family_index = family_vals.index(df_top["family"].iloc[0])
            item_id_index = item_id_vals.index(df_top["item_id"].iloc[0])

            channel_choice = st.selectbox("Channel", channel_vals, index=channel_index)
            family_choice = st.selectbox("Family", family_vals, index=family_index)
            item_id_choice = st.selectbox("Item ID", item_id_vals, index=item_id_index)

        with col2:
            results_mask = \
                make_mask(df_results, channel_choice, family_choice, item_id_choice)
            pred_mask = \
                make_mask(df_pred, channel_choice, family_choice, item_id_choice)

            df_plot = df_pred[pred_mask]

            st.markdown("#### Chart")

            if len(df_plot) > 0:

                # display the line chart
                #fig = px.line(df_plot, x="timestamp", y="demand", color="type")

                y = df_plot.query("type == 'actual'")["demand"]
                y_ts = df_plot.query("type == 'actual'")["timestamp"]

                yp = df_plot.query("type == 'fcast'")["demand"]
                yp_ts = df_plot.query("type == 'fcast'")["timestamp"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_ts, y=y, mode='lines+markers', name="actual",
                    marker=dict(size=4)
                ))
                fig.add_trace(go.Scatter(
                    x=yp_ts, y=yp, mode='lines+markers', name="forecast", line_dash="dot",
                    marker=dict(size=4)
                ))
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
                        margin={"t": 30, "b": 0, "r": 0, "l": 40},
                    height=290,
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor":"center", "x": 0.5}
                )
                st.plotly_chart(fig, use_container_width=True)

#       report_f = state.df_results.to_csv(index=False)
#       report_b64 = base64.b64encode(report_f.encode()).decode()
#       report_fn = f"{state.uploaded_file.name}_report.csv"

#       href_html = f"""
#       - <a href='data:file/csv;base64,{forecast_b64}' download='{forecast_fn}'>{forecast_fn}</a>
#       - <a href='data:file/csv;base64,{report_b64}' download='{report_fn}'>{report_fn}</a>
#       """

#       st.subheader("Downloads")
#       dl_button = st.button("Download Forecast")

#       if dl_button:
#           f = state.df_pred.to_csv(index=False)
#           forecast_b64 = base64.b64encode(f.encode()).decode()
#           forecast_fn = f"{state.uploaded_file.name}_forecast.csv"

#           report_f = state.df_results.to_csv(index=False)
#           report_b64 = base64.b64encode(report_f.encode()).decode()
#           report_fn = f"{state.uploaded_file.name}_report.csv"

#           href_html = f"""
#           - <a href="javascript:void(0)" onclick="location.href='data:file/csv;base64,{report_b64}'" download='{report_fn}'>{report_fn}</a>
#           """

#           st.markdown(href_html, unsafe_allow_html=True)

#       - <a href='data:file/csv;base64,{report_b64}' download='{report_fn}'>{report_fn}</a>
#       """



#       with col3:
#           st.markdown("#### Summary")

#       st.subheader("DEBUGGING OUTPUT")
#       st.text("df_results[mask]")
#       st.dataframe(df_pred[pred_mask])

#        if len(df) > 0:
#            #
#            # Display the chart
#            #
#            fig = px.line(df, x="timestamp", y="demand", color="type")
#
#    #       #fig.add_vline(x="2016-01-01", line_width=1, line_dash="dot")
#    #       #fig.add_vline(x="2017-01-01", line_width=1, line_dash="dot")
#
#            fig.update_layout(
#                xaxis={
#                    "showgrid": True,
#                    "gridcolor": "lightgrey",
#                },
#                yaxis={
#                    "showgrid": True,
#                    "gridcolor": "lightgrey",
#                }
#            )
#
#            st.plotly_chart(fig, use_container_width=True)
#

    return


def display_progress(wait_for):
    """
    """

    # display progress of the futures
    pbar = stqdm(desc="Progress", total=len(wait_for))
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

    #pbar.close()

    return


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    state = get_state()

    #st.set_page_config(layout="wide")
    #st.beta_container()

    pages = {
        "Create Forecast": page_upload_file,
        "View Report": page_view_report
    }

    st.sidebar.title("Simple Forecast Solution")
    page = st.sidebar.selectbox("Select Page", ["Create Forecast", "View Report"])
    pages[page](state)

    state.sync()

    st.stop()
