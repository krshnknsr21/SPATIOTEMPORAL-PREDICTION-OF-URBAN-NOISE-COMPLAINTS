"""
CANSF-ZIP – Construction-Aware Noise Spatiotemporal Forecasting (ZIP level)
===========================================================================

End-to-end data mining system to:
1. Load & preprocess NYC 311 noise complaints (ZIP-level)
2. Enrich with DOB construction permit data (ZIP-level)
3. Engineer temporal + lag + construction features
4. Train baselines and a novel two-stage model (CANSF)
5. Evaluate with different metrics & visualizations
6. Forecast hourly ZIP-level noise for 2026 + key influencing factors
"""

# ============================================================
#  0. Imports & Config
# ============================================================
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ---------------- CONFIG: tuned to your REAL columns ----------------
CONFIG = {
    # 311 noise dataset (all years or subset)
    "NYC311_PATH": "21-22_Test.csv",

    # DOB construction permit file (subset is fine)
    "DOB_PERMIT_PATH": "dob_Permit_Test.xlsx",

    # 311 column names in CSV
    "COL_CREATED_DATE": "created_date",
    "COL_BOROUGH": "borough",
    "COL_COMPLAINT_TYPE": "complaint_type",
    "COL_ZIP": "incident_zip",

    # DOB permit columns (from your screenshot)
    "DOB_COL_BORO": "BOROUGH",
    "DOB_COL_ZIP": "Zip Code",
    "DOB_COL_ISSUE_DATE": "Issuance Date",
    "DOB_COL_EXPIRY_DATE": "Expiration Date",
    "DOB_COL_JOB_TYPE": "Job Type",      # we treat Job Type as construction type
    "DOB_COL_AFTER_HOURS": None,         # no explicit after-hours field

    # General
    "NOISE_KEYWORD": "Noise",            # filter complaint_type containing this
    "TRAIN_END_DATE": "2021-01-31",
    "TEST_END_DATE": "2021-02-08",
    "FORECAST_START_2026": "2026-01-01",
    "FORECAST_END_2026": "2026-12-31",

    # Lag settings (in hours)
    "LAGS": [1, 24],

    # Construction intensity threshold for "high-construction" analysis
    "HIGH_CONSTR_THRESHOLD": 5.0,
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================
#  1. Utility Functions
# ============================================================
def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def safe_parse_datetime(series):
    return pd.to_datetime(series, errors="coerce")

from sklearn.metrics import root_mean_squared_error

def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


# ============================================================
#  2. Load & Preprocess NYC 311 Noise Data (ZIP + hour)
# ============================================================
def load_and_prepare_311(config: dict) -> pd.DataFrame:
    print_section("Loading & preprocessing NYC 311 noise data (ZIP level)")

    path = config["NYC311_PATH"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"311 file not found at {path}")

    df = pd.read_csv(path, low_memory=False)
    print(f"Raw 311 shape: {df.shape}")

    # normalize column names to lower-case
    df.columns = [c.strip().lower() for c in df.columns]

    col_created = config["COL_CREATED_DATE"].lower()
    col_boro = config["COL_BOROUGH"].lower()
    col_type = config["COL_COMPLAINT_TYPE"].lower()
    col_zip = config["COL_ZIP"].lower()

    for col in [col_created, col_boro, col_type, col_zip]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in 311 data.")

    # filter noise complaints
    noise_keyword = config["NOISE_KEYWORD"].lower()
    df = df[df[col_type].str.contains(noise_keyword, case=False, na=False)].copy()
    print(f"After filtering to noise complaints: {df.shape}")

    # drop rows without key fields
    df = df.dropna(subset=[col_created, col_boro, col_zip])

    # parse datetime
    df["created_dt"] = safe_parse_datetime(df[col_created])
    df = df.dropna(subset=["created_dt"])

    # normalize borough, zip
    df["borough"] = df[col_boro].astype(str).str.strip().str.upper()
    df["zip"] = df[col_zip].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)

    # aggregate to hourly ZIP counts per borough
    df["created_dt_hour"] = df["created_dt"].dt.floor("h")
    agg = (
        df.groupby(["borough", "zip", "created_dt_hour"])
        .size()
        .reset_index(name="noise_count")
    )
    agg = agg.sort_values(["borough", "zip", "created_dt_hour"]).reset_index(drop=True)

    print(f"Aggregated hourly ZIP data shape: {agg.shape}")
    print(agg.head())

    return agg


# ============================================================
#  3. Load & Preprocess DOB Construction Data (ZIP + day)
# ============================================================
def load_and_prepare_dob(config: dict) -> pd.DataFrame:
    print_section("Loading & preprocessing DOB construction permit data (ZIP level)")

    path = config["DOB_PERMIT_PATH"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"DOB permits file not found at {path}")

    dob = pd.read_excel(path)
    print(f"Raw DOB shape: {dob.shape}")

    dob.columns = [c.strip().upper() for c in dob.columns]

    col_boro = config["DOB_COL_BORO"].upper()
    col_zip = config["DOB_COL_ZIP"].upper()
    col_issue = config["DOB_COL_ISSUE_DATE"].upper()
    col_exp = config["DOB_COL_EXPIRY_DATE"].upper()
    col_job = config["DOB_COL_JOB_TYPE"].upper()
    col_ah = config["DOB_COL_AFTER_HOURS"]
    if col_ah is not None:
        col_ah = col_ah.upper()

    for col in [col_boro, col_zip, col_issue]:
        if col not in dob.columns:
            raise KeyError(f"Required DOB column '{col}' not found.")

    # basic cleaning
    dob = dob.dropna(subset=[col_boro, col_zip, col_issue]).copy()
    dob["BOROUGH"] = dob[col_boro].astype(str).str.strip().str.upper()
    dob["ZIP"] = dob[col_zip].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)

    dob["ISSUE_DT"] = safe_parse_datetime(dob[col_issue])
    dob["EXPIRY_DT"] = safe_parse_datetime(dob[col_exp]) if col_exp in dob.columns else pd.NaT

    # if expiry missing, assume 90 days after issue
    dob["EXPIRY_DT"] = dob["EXPIRY_DT"].fillna(dob["ISSUE_DT"] + pd.Timedelta(days=90))

    # clip extremely long permits (max 2 years)
    max_days = 365 * 2
    dob["EXPIRY_DT"] = dob[["ISSUE_DT", "EXPIRY_DT"]].apply(
        lambda row: min(row["EXPIRY_DT"], row["ISSUE_DT"] + pd.Timedelta(days=max_days)),
        axis=1,
    )

    # Job type categories – heavy construction if job type mentions NEW / DEMOL
    if col_job in dob.columns:
        dob["JOB_TYPE_CAT"] = dob[col_job].astype(str).str.upper()
        dob["HEAVY_JOB"] = dob["JOB_TYPE_CAT"].str.contains("NEW|DEMOL", regex=True)
    else:
        dob["HEAVY_JOB"] = False

    # After-hours flag – not available, assume False
    if col_ah is not None and col_ah in dob.columns:
        dob["AFTER_HOURS"] = dob[col_ah].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])
    else:
        dob["AFTER_HOURS"] = False

    print("DOB sample:")
    print(dob[["BOROUGH", "ZIP", "ISSUE_DT", "EXPIRY_DT", "HEAVY_JOB", "AFTER_HOURS"]].head())

    # Expand permits to daily borough+ZIP activity
    print("\nExpanding DOB permits to daily ZIP activity (may take some time for large files)...")

    records = []
    for _, row in dob.iterrows():
        start = row["ISSUE_DT"].normalize()
        end = row["EXPIRY_DT"].normalize()
        if pd.isna(start) or pd.isna(end) or start > end:
            continue
        if (end - start).days > 400:
            end = start + pd.Timedelta(days=400)
        date_range = pd.date_range(start, end, freq="D")
        for d in date_range:
            records.append(
                {
                    "borough": row["BOROUGH"],
                    "zip": row["ZIP"],
                    "date": d,
                    "heavy_job": bool(row["HEAVY_JOB"]),
                    "after_hours": bool(row["AFTER_HOURS"]),
                }
            )

    daily = pd.DataFrame.from_records(records)
    if daily.empty:
        raise ValueError("Daily DOB expansion ended up empty – check your DOB data.")

    daily_agg = (
        daily.groupby(["borough", "zip", "date"])
        .agg(
            active_permits=("zip", "size"),
            heavy_job_count=("heavy_job", "sum"),
            ahv_count=("after_hours", "sum"),
        )
        .reset_index()
    )

    daily_agg["construction_intensity_index"] = (
        1.0 * daily_agg["active_permits"]
        + 1.5 * daily_agg["heavy_job_count"]
        + 2.0 * daily_agg["ahv_count"]
    )

    print(f"Daily DOB ZIP activity shape: {daily_agg.shape}")
    print(daily_agg.head())

    return daily_agg


# ============================================================
#  4. Combine 311 & DOB, Feature Engineering
# ============================================================
def engineer_features(agg_311: pd.DataFrame, dob_daily: pd.DataFrame, config: dict) -> pd.DataFrame:
    print_section("Feature engineering: temporal, lag, construction (ZIP level)")

    df = agg_311.copy()
    df["date"] = df["created_dt_hour"].dt.normalize()

    # temporal features
    df["hour"] = df["created_dt_hour"].dt.hour
    df["dayofweek"] = df["created_dt_hour"].dt.dayofweek
    df["month"] = df["created_dt_hour"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # sort before lags
    df = df.sort_values(["borough", "zip", "created_dt_hour"]).reset_index(drop=True)

    # lag features per borough+zip
    for lag in config["LAGS"]:
        df[f"lag_{lag}"] = (
            df.groupby(["borough", "zip"])["noise_count"]
            .shift(lag)
        )

    # merge in construction features (borough + ZIP + date)
    dob_daily["date"] = pd.to_datetime(dob_daily["date"])
    dob_daily["borough"] = dob_daily["borough"].astype(str).str.upper()
    dob_daily["zip"] = dob_daily["zip"].astype(str).str.zfill(5)

    df = df.merge(
        dob_daily,
        on=["borough", "zip", "date"],
        how="left",
    )

    # fill missing construction with zeros (no active permits)
    constr_cols = ["active_permits", "heavy_job_count", "ahv_count", "construction_intensity_index"]
    for c in constr_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # drop rows with missing lags
    df = df.dropna(subset=[f"lag_{lag}" for lag in config["LAGS"]])

    print(f"Final feature table shape: {df.shape}")
    print(df.head())

    return df


# ============================================================
#  5. Train/Test Split
# ============================================================
def train_test_split_time(df: pd.DataFrame, config: dict):
    print_section("Train / Test split by time")

    train_end = pd.to_datetime(config["TRAIN_END_DATE"])
    test_end = pd.to_datetime(config["TEST_END_DATE"])

    df = df.sort_values("created_dt_hour")

    train = df[df["created_dt_hour"] <= train_end].copy()
    test = df[(df["created_dt_hour"] > train_end) & (df["created_dt_hour"] <= test_end)].copy()

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test


# ============================================================
#  6. Baseline Models
# ============================================================
def build_feature_sets(df: pd.DataFrame):
    temporal_feats = ["hour", "dayofweek", "month", "is_weekend"]
    lag_feats = [c for c in df.columns if c.startswith("lag_")]
    constr_feats = [
        c for c in ["active_permits", "heavy_job_count", "ahv_count", "construction_intensity_index"]
        if c in df.columns
    ]
    return temporal_feats, lag_feats, constr_feats


def evaluate_model(name, y_true, y_pred, extra_mask=None):
    if extra_mask is not None:
        y_true = y_true[extra_mask]
        y_pred = y_pred[extra_mask]
        if len(y_true) == 0:
            return {"model": name, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "n": 0}

    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "n": len(y_true),
    }


def run_baselines(train, test):
    print_section("Training baselines")

    temporal_feats, lag_feats, _ = build_feature_sets(train)
    target = "noise_count"
    results = []
    preds = {}

    y_true = test[target].values

    # Baseline 0 – historical mean by (ZIP, hour)
    print("Baseline: historical mean by (zip, hour)")
    hist_means = (
        train.groupby(["zip", "hour"])[target]
        .mean()
        .rename("hist_mean")
        .reset_index()
    )
    test0 = test.merge(hist_means, on=["zip", "hour"], how="left")
    test0["hist_mean"] = test0["hist_mean"].fillna(test[target].mean())
    y_pred = test0["hist_mean"].values
    results.append(evaluate_model("HistMean_zip_hour", y_true, y_pred))
    preds["HistMean_zip_hour"] = y_pred

    # Baseline 1 – Linear Regression (temporal only)
    print("Baseline: Linear Regression (temporal)")
    lr = LinearRegression()
    X_train = train[temporal_feats]
    X_test = test[temporal_feats]
    lr.fit(X_train, train[target])
    y_pred = lr.predict(X_test)
    results.append(evaluate_model("LinearRegression_temporal", y_true, y_pred))
    preds["LinearRegression_temporal"] = y_pred

    # Baseline 2 – Random Forest (temporal + lags)
    print("Baseline: RandomForest (temporal + lags)")
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=None,
    )
    X_train = train[temporal_feats + lag_feats]
    X_test = test[temporal_feats + lag_feats]
    rf.fit(X_train, train[target])
    y_pred = rf.predict(X_test)
    results.append(evaluate_model("RandomForest_time_lags", y_true, y_pred))
    preds["RandomForest_time_lags"] = y_pred

    baseline_models = {
        "HistMean": {"preds": preds["HistMean_zip_hour"], "obj": None},
        "LR": {"preds": preds["LinearRegression_temporal"], "obj": lr},
        "RF": {"preds": preds["RandomForest_time_lags"], "obj": rf},
    }

    return results, baseline_models


# ============================================================
#  7. Novel Two-Stage Model: CANSF (ZIP level)
# ============================================================
def run_cansf(train, test):
    print_section("Training CANSF (two-stage construction-aware model, ZIP level)")

    temporal_feats, lag_feats, constr_feats = build_feature_sets(train)
    target = "noise_count"

    # Stage 1 – baseline XGBoost on temporal + lags
    print("Stage 1: XGBoost on temporal + lags (no construction)")
    base_features = temporal_feats + lag_feats

    base_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    base_model.fit(train[base_features], train[target])

    train_base_pred = base_model.predict(train[base_features])
    test_base_pred = base_model.predict(test[base_features])

    # residuals
    train_resid = train[target] - train_base_pred
    test_resid_true = test[target] - test_base_pred  # optional analysis

    # Stage 2 – residual model on construction + temporal features
    print("Stage 2: XGBoost residual model on construction + temporal features")
    if len(constr_feats) == 0:
        raise ValueError("No construction features available to train residual model.")

    resid_features = temporal_feats + constr_feats

    resid_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    resid_model.fit(train[resid_features], train_resid)

    train_resid_pred = resid_model.predict(train[resid_features])
    test_resid_pred = resid_model.predict(test[resid_features])

    train_final_pred = train_base_pred + train_resid_pred
    test_final_pred = test_base_pred + test_resid_pred

    # evaluation
    y_true_train = train[target].values
    y_true_test = test[target].values

    overall_train = evaluate_model("CANSF_train", y_true_train, train_final_pred)
    overall_test = evaluate_model("CANSF_test", y_true_test, test_final_pred)

    print("CANSF Train:", overall_train)
    print("CANSF Test:", overall_test)

    models = {
        "base_model": base_model,
        "resid_model": resid_model,
        "base_features": base_features,
        "resid_features": resid_features,
    }

    evals = {
        "train": overall_train,
        "test": overall_test,
        "test_base_pred": test_base_pred,
        "test_final_pred": test_final_pred,
        "test_resid_true": test_resid_true,
        "test_resid_pred": test_resid_pred,
    }

    return models, evals


# ============================================================
#  8. Evaluation: High vs Low Construction, Visualizations
# ============================================================
def analyze_and_visualize(train, test, baseline_results, baseline_models, cansf_evals, config):
    print_section("Comprehensive evaluation & visualizations")

    # merge metrics
    all_results = list(baseline_results)
    all_results.append({
        "model": "CANSF_two_stage",
        "MAE": cansf_evals["test"]["MAE"],
        "RMSE": cansf_evals["test"]["RMSE"],
        "R2": cansf_evals["test"]["R2"],
        "n": cansf_evals["test"]["n"],
    })
    results_df = pd.DataFrame(all_results)
    print("\nOverall metrics (test set):")
    print(results_df)

    # High vs low construction performance
    high_thresh = config["HIGH_CONSTR_THRESHOLD"]
    high_mask = test["construction_intensity_index"] >= high_thresh
    low_mask = ~high_mask

    print(f"\nHigh-construction hours (index >= {high_thresh}): {high_mask.sum()} rows")
    print(f"Low-construction hours: {low_mask.sum()} rows")

    y_true = test["noise_count"].values
    hc_results = []

    rf_pred = baseline_models["RF"]["preds"]
    hc_results.append(evaluate_model("RF_highConstr", y_true, rf_pred, extra_mask=high_mask))
    hc_results.append(evaluate_model("RF_lowConstr", y_true, rf_pred, extra_mask=low_mask))

    cansf_pred = cansf_evals["test_final_pred"]
    hc_results.append(evaluate_model("CANSF_highConstr", y_true, cansf_pred, extra_mask=high_mask))
    hc_results.append(evaluate_model("CANSF_lowConstr", y_true, cansf_pred, extra_mask=low_mask))

    hc_df = pd.DataFrame(hc_results)
    print("\nHigh vs Low construction performance:")
    print(hc_df)

    os.makedirs("figures", exist_ok=True)

    # 1. time series for one ZIP (most active in test)
    sample_zip = test["zip"].value_counts().index[0]
    test_zip = test[test["zip"] == sample_zip].copy().sort_values("created_dt_hour")
    idx = test_zip.index

    plt.figure(figsize=(14, 5))
    plt.plot(test_zip["created_dt_hour"], test_zip["noise_count"], label="Actual", linewidth=1)
    plt.plot(test_zip["created_dt_hour"], baseline_models["RF"]["preds"][idx], label="RF_pred", alpha=0.7)
    plt.plot(test_zip["created_dt_hour"], cansf_evals["test_final_pred"][idx], label="CANSF_pred", alpha=0.7)
    plt.title(f"Time Series – Actual vs RF vs CANSF (ZIP {sample_zip})")
    plt.xlabel("Time")
    plt.ylabel("Noise complaints")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/time_series_zip_example.png", dpi=150)
    plt.close()

    # 2. scatter actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(test["noise_count"], baseline_models["RF"]["preds"], alpha=0.3, label="RF")
    plt.scatter(test["noise_count"], cansf_evals["test_final_pred"], alpha=0.3, label="CANSF")
    max_val = max(test["noise_count"].max(), cansf_evals["test_final_pred"].max())
    plt.plot([0, max_val], [0, max_val], "k--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (RF vs CANSF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/actual_vs_pred_rf_vs_cansf.png", dpi=150)
    plt.close()

    print("\nSaved figures in ./figures/")
    return results_df, hc_df


def plot_feature_importances(models_dict):
    base_model = models_dict["base_model"]
    resid_model = models_dict["resid_model"]
    base_feats = models_dict["base_features"]
    resid_feats = models_dict["resid_features"]

    os.makedirs("figures", exist_ok=True)

    if hasattr(base_model, "feature_importances_"):
        fi = pd.Series(base_model.feature_importances_, index=base_feats).sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        fi.head(15).plot(kind="bar")
        plt.title("Stage 1 – Feature Importances (XGBoost base)")
        plt.tight_layout()
        plt.savefig("figures/feature_importance_stage1_base.png", dpi=150)
        plt.close()

    if hasattr(resid_model, "feature_importances_"):
        fi = pd.Series(resid_model.feature_importances_, index=resid_feats).sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        fi.head(15).plot(kind="bar")
        plt.title("Stage 2 – Feature Importances (XGBoost residual, construction-aware)")
        plt.tight_layout()
        plt.savefig("figures/feature_importance_stage2_resid.png", dpi=150)
        plt.close()


# ============================================================
#  9. Build 2026 ZIP Feature Grid & Forecast
# ============================================================
def build_2026_feature_grid(dob_daily, config):
    print_section("Building 2026 ZIP feature grid")

    start = pd.to_datetime(config["FORECAST_START_2026"])
    end = pd.to_datetime(config["FORECAST_END_2026"])
    hours = pd.date_range(start, end + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq="h")

    zips = dob_daily["zip"].unique()
    boroughs = dob_daily[["zip", "borough"]].drop_duplicates()

    grid = pd.MultiIndex.from_product([zips, hours], names=["zip", "created_dt_hour"]).to_frame(index=False)
    grid = grid.merge(boroughs, on="zip", how="left")

    grid["date"] = grid["created_dt_hour"].dt.normalize()
    grid["hour"] = grid["created_dt_hour"].dt.hour
    grid["dayofweek"] = grid["created_dt_hour"].dt.dayofweek
    grid["month"] = grid["created_dt_hour"].dt.month
    grid["is_weekend"] = grid["dayofweek"].isin([5, 6]).astype(int)

    dob_daily["date"] = pd.to_datetime(dob_daily["date"])
    grid = grid.merge(
        dob_daily,
        on=["borough", "zip", "date"],
        how="left",
    )
    for c in ["active_permits", "heavy_job_count", "ahv_count", "construction_intensity_index"]:
        if c in grid.columns:
            grid[c] = grid[c].fillna(0.0)

    print(f"2026 grid shape: {grid.shape}")
    return grid


def forecast_2026(grid_2026, models_dict):
    print_section("Forecasting 2026 with CANSF (ZIP level)")

    base_model = models_dict["base_model"]
    resid_model = models_dict["resid_model"]
    base_feats = models_dict["base_features"]
    resid_feats = models_dict["resid_features"]

    # -------------------- FIX LAG FEATURES --------------------
    # XGBoost requires lag_1 and lag_24 because they were used in training.
    for lag in [1, 24]:
        lag_col = f"lag_{lag}"
        if lag_col not in grid_2026.columns:
            grid_2026[lag_col] = 0.0

    # drop lag features for forward-only forecast
    # KEEP all base features (including lag_1 and lag_24)  
    base_feats_forecast = base_feats  

# Ensure grid includes the lag columns  
    for lag in [1, 24]:
        lag_col = f"lag_{lag}"
        if lag_col not in grid_2026.columns:
            grid_2026[lag_col] = 0.0

    resid_feats_forecast = [f for f in resid_feats if f in grid_2026.columns]

    print("Base features used for 2026:", base_feats_forecast)
    print("Residual features used for 2026:", resid_feats_forecast)

    base_pred_2026 = base_model.predict(grid_2026[base_feats_forecast])
    resid_pred_2026 = resid_model.predict(grid_2026[resid_feats_forecast])
    final_pred_2026 = base_pred_2026 + resid_pred_2026

    grid_2026["pred_noise_count"] = final_pred_2026

    summary = (
        grid_2026.groupby(["borough", "zip"])["pred_noise_count"]
        .agg(["mean", "max", "sum"])
        .reset_index()
        .rename(columns={"mean": "avg_hourly", "max": "peak_hourly", "sum": "total_2026"})
    )

    print("\n2026 forecast summary (top 10 ZIPs by total complaints):")
    print(summary.sort_values("total_2026", ascending=False).head(10))

    grid_2026.to_csv("predictions_2026_zip_hourly.csv", index=False)
    summary.to_csv("predictions_2026_summary_by_zip.csv", index=False)
    print("\nSaved 2026 predictions to:")
    print("- predictions_2026_zip_hourly.csv")
    print("- predictions_2026_summary_by_zip.csv")

    return grid_2026, summary


# ============================================================
#  10. Main
# ============================================================
def main():
    agg_311 = load_and_prepare_311(CONFIG)
    dob_daily = load_and_prepare_dob(CONFIG)

    df = engineer_features(agg_311, dob_daily, CONFIG)
    train, test = train_test_split_time(df, CONFIG)

    baseline_results, baseline_models = run_baselines(train, test)
    cansf_models, cansf_evals = run_cansf(train, test)

    results_df, hc_df = analyze_and_visualize(train, test, baseline_results, baseline_models, cansf_evals, CONFIG)
    results_df.to_csv("model_comparison_metrics.csv", index=False)
    hc_df.to_csv("model_high_low_construction_metrics.csv", index=False)
    print("\nSaved metrics to:")
    print("- model_comparison_metrics.csv")
    print("- model_high_low_construction_metrics.csv")

    plot_feature_importances(cansf_models)

    grid_2026 = build_2026_feature_grid(dob_daily, CONFIG)
    forecast_2026(grid_2026, cansf_models)

    print("\nDone  – CANSF-ZIP pipeline executed successfully.")


if __name__ == "__main__":
    main()
