import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)

def feature_engineering(path_to_dataset, path_to_weather, validation : bool): 

    data = pd.read_csv(path_to_dataset)
    weather = pd.read_csv(path_to_weather)

    # Merging data & weather 
    df = data.merge(weather, on="DATETIME", how="left")

    # Extracting time features
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["hour"] = df["DATETIME"].dt.hour
    df["dow"] = df["DATETIME"].dt.dayofweek
    df["month"] = df["DATETIME"].dt.month

    # Adding time cyclical features 
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]   = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["dow"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    # Adding time event flags 
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_lunch"]  = df["hour"].between(12, 14).astype(int)
    df["is_dinner"] = df["hour"].between(18, 20).astype(int)
    df["is_opening_peak"] = df["hour"].between(9,11).astype(int)
    df["is_closing_peak"] = df["hour"].between(17,19).astype(int)


    # Quantifying park-wide congestion 
    df = df.sort_values(["DATETIME", "ENTITY_DESCRIPTION_SHORT"]).copy()
    grp_t = df.groupby("DATETIME", group_keys=False)

    df["park_cwt_mean_t"]   = grp_t["CURRENT_WAIT_TIME"].transform("mean")
    df["park_cwt_median_t"] = grp_t["CURRENT_WAIT_TIME"].transform("median")
    df["park_cwt_std_t"]    = grp_t["CURRENT_WAIT_TIME"].transform("std").fillna(0)

    # Percentile rank of the ride's current wait at that timestamp
    df["cwt_rank_within_time"] = grp_t["CURRENT_WAIT_TIME"].transform(lambda s: s.rank(pct=True))

    # Deviation and z-score vs park mean (robust signals of relative congestion)
    df["cwt_dev_from_mean"]  = df["CURRENT_WAIT_TIME"] - df["park_cwt_mean_t"]
    df["cwt_zscore_in_time"] = df["cwt_dev_from_mean"] / (df["park_cwt_std_t"] + 1e-6)

    # (Optional) capacity context for the same moment
    df["park_capacity_sum_t"]   = grp_t["ADJUST_CAPACITY"].transform("sum")
    df["ride_share_of_capacity_t"] = df["ADJUST_CAPACITY"] / (df["park_capacity_sum_t"] + 1e-6)

    # Quantifying interactions & momentum 
    df["cap_times_cwt"] = df["ADJUST_CAPACITY"] * df["CURRENT_WAIT_TIME"]
    df["downtime_flag"] = (df["DOWNTIME"] > 0).astype(int)

    # Dealing with sparse columns 
    # Dropping TIME_TO_PARADE_2 since missing 83‰ of values 
    for col in ["TIME_TO_PARADE_2"]: 
        if col in df.columns: df.drop(columns=[col], inplace=True)
    # Replacing missing TIME_TO_PARADE_1 and TIME_TO_NIGHT_SHOW to the average time 
    for col in ["TIME_TO_PARADE_1","TIME_TO_NIGHT_SHOW"]:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
    # Replacing missing snow_1h and rain_1h to 0 value 
    for col in ["snow_1h","rain_1h"]:
        if col in df.columns: df[col] = df[col].fillna(0)

    # final median fill for lags at sequence starts
    num_cols_all = df.select_dtypes(include=[np.number]).columns
    df[num_cols_all] = df[num_cols_all].fillna(df[num_cols_all].median())

    # Dropping original time features 
    df = df.drop(columns=["DATETIME"]) 
    # One-hot encoding attraction names into labels 
    df["ENTITY_DESCRIPTION_SHORT"] = LabelEncoder().fit_transform(df["ENTITY_DESCRIPTION_SHORT"])

    if not validation:
        y = df["WAIT_TIME_IN_2H"]
        X = df.drop(columns=["WAIT_TIME_IN_2H"])
        return X, y
    else:
        if "WAIT_TIME_IN_2H" in df.columns:
            df = df.drop(columns=["WAIT_TIME_IN_2H"])
        return df 
    
