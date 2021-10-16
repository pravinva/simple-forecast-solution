import afa
import pytest
import pandas as pd
import pandera as pa

from darts.datasets import AirPassengersDataset, TemperatureDataset
from afa import load_data, run_cv_select
from afa.core import impute_dates, resample


@pytest.fixture
def df_monthly():

    df = AirPassengersDataset().load().pd_dataframe()
    df["timestamp"] = df.index.strftime("%Y-%m-%d")
    df["channel"] = "ch01"
    df["family"] = "family01"
    df["item_id"] = "item01"
    df["demand"] = df["#Passengers"]
    df.drop("#Passengers", axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.columns.name = None

    return df


@pytest.fixture
def df_daily():
    df1 = TemperatureDataset() \
            .load() \
            .pd_dataframe() \
            .rename({"Daily minimum temperatures": "demand"}, axis=1)
    df1["timestamp"] = df1.index
    df1["channel"] = "channel"
    df1["item_id"] = "item_id"
    df1["family"] = "family"
    df1["demand"] = df1["demand"].fillna(method="ffill")
    df1.reset_index(drop=True, inplace=True)
    df1.columns.name = None

    return df1


def test_impute_dates(df_daily):
    df1 = df_daily.copy()
    
    # randomly drop 10% of rows, keep the first and last rows
    df2 = pd.concat([
        df1.iloc[[0]],
        df1.iloc[1:-1].sample(frac=0.1, random_state=42),
        df1.iloc[[-1]]]) \
    .sort_index()

    df2.set_index("timestamp", inplace=True)
    df2.index = pd.DatetimeIndex(df2.index)

    # impute the missing dates
    df3 = impute_dates(df2, "D")

    assert(df3.shape[0] == 3652)
    
    return


def test_resample(df_daily):
    df1 = df_daily.copy()
    df2 = load_data(df1)
    df3 = resample(df2, "W-MON")

    assert df3.shape == (522, 4)
    
    return


def test_load_data(df_monthly):
    df1 = df_monthly.copy()
    df2 = df_monthly.copy()

    df = load_data(df1, impute_freq=None)

    assert df.shape[0] == df1.shape[0]

    # Inject some null values and test internal panderea dataframe validation
    df2.at[df1.index[0],"demand"] = None
    df2.at[df1.index[10],"item_id"] = None
    df2.at[df1.index[20],"family"] = None

    try:
        df = load_data(df2, impute_freq=None)
    except pa.errors.SchemaErrors as err:
        df_fail = err.failure_cases

        assert df_fail.shape[0] == 3
        assert df_fail["column"].tolist() == ["family", "item_id", "demand"]
        assert df_fail["check"].tolist() == ["not_nullable"] * 3
        assert df_fail["index"].tolist() == [20, 10, 0]
    
    return


def test_run_cv_select(df_monthly):
    df = df_monthly.copy()
    
    df_pred, df_results = run_cv_select(df, 6, "MS", cv_stride=1)

    return