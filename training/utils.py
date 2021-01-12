from tqdm import tqdm
import numpy as np
import pandas as pd
import time

def reduce_mem(df):
    """ Reduce memory """
    starttime = time.time()
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(
        "-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min".format(
            end_mem,
            100 * (start_mem - end_mem) / start_mem,
            (time.time() - starttime) / 60,
        )
    )
    return df


def unique_count(df, features):
    """ perform a unique count for categorical features"""
    for f in tqdm(features):
        map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
        df[f] = df[f].map(map_dict)
        df[f + "_count"] = df[f].map(df[f].value_counts())
    df = reduce_mem(df)
    return df


def get_click_histories(df, col, windows):
    """ actions past a certain time window """
    # total clicks past n days groupby a certain feature
    clicks = (
        df[[col, "pt_d", "label"]]
        .groupby([col, "pt_d"], as_index=False)["label"]
        .agg({f"{col}_prev_{windows}_day_click_count": "sum"})
    )
    clicks["pt_d"] += windows
    df = df.merge(clicks, on=[col, "pt_d"], how="left")
    df[f"{col}_prev_{windows}_day_click_count"] = df[
        f"{col}_prev_{windows}_day_click_count"
    ].fillna(0)

    # total impressions
    impressions = (
        df[[col, "pt_d", "label"]]
        .groupby([col, "pt_d"], as_index=False)["label"]
        .agg({f"{col}_prev_{windows}_day_count": "count"})
    )
    impressions["pt_d"] += windows
    df = df.merge(impressions, on=[col, "pt_d"], how="left")
    df[f"{col}_prev_{windows}_day_count"] = df[
        f"{col}_prev_{windows}_day_count"
    ].fillna(0)

    # total click through rates
    df[f"{col}_prev_{windows}_day_ctr"] = df[
        f"{col}_prev_{windows}_day_click_count"
    ] / (
        df[f"{col}_prev_{windows}_day_count"]
        + df[f"{col}_prev_{windows}_day_count"].mean()
    )

    # drop temp features
    df.drop(
        [f"{col}_prev_{windows}_day_click_count", f"{col}_prev_{windows}_day_count"],
        inplace=True,
        axis=1,
    )
    del clicks, impressions
    return df
