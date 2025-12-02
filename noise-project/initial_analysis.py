"""
Spatiotemporal Prediction of NYC 311 Noise Complaints – Checkpoint 2
Author: Group 20 (Aman Mukherjee, Atharva Deshpande, Krishna Kansara)

This script demonstrates:
1. Data loading & cleaning
2. Feature engineering (temporal + lag)
3. Train/test split
4. Model training & evaluation
5. Visualization of results
"""

# ============================================================
#  Imports
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
#  1. Load Data
# ============================================================
print("Loading CSV ...")
# df = pd.read_csv("nyc311_noise_2025 - Copy.csv", low_memory=False)
df = pd.read_csv("nyc311_noise_all_years.csv", low_memory=False)

print("\nInitial shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ============================================================
#  2. Data Cleaning
# ============================================================
print("\nCleaning data ...")

# Convert datetimes
df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

# Drop missing important fields
df = df.dropna(subset=["created_date", "latitude", "longitude", "borough"])

# Keep only noise-related complaints (defensive filter)
df = df[df["complaint_type"].str.contains("Noise", case=False, na=False)]

# Drop duplicate complaint IDs
df = df.drop_duplicates(subset=["unique_key"])

print("After cleaning:", df.shape)
print("Date range:", df["created_date"].min(), "to", df["created_date"].max())
print("Borough counts:\n", df["borough"].value_counts())

# ============================================================
#  3. Feature Engineering
# ============================================================
print("\nFeature engineering ...")

df["hour"] = df["created_date"].dt.hour
df["dayofweek"] = df["created_date"].dt.dayofweek   # 0=Mon
df["month"] = df["created_date"].dt.month
df["weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# Aggregate to hourly counts per borough
agg = (
    df.groupby(["borough", pd.Grouper(key="created_date", freq="H")])
      ["unique_key"].count().reset_index(name="count")
)
agg = agg.sort_values(["borough", "created_date"])

# Create lag features (previous hour, previous day)
agg["lag1"] = agg.groupby("borough")["count"].shift(1)
agg["lag24"] = agg.groupby("borough")["count"].shift(24)
agg = agg.dropna()

# Add temporal features back
agg["hour"] = agg["created_date"].dt.hour
agg["dayofweek"] = agg["created_date"].dt.dayofweek
agg["month"] = agg["created_date"].dt.month
agg["weekend"] = agg["dayofweek"].isin([5, 6]).astype(int)

print("Aggregated data shape:", agg.shape)
print(agg.head())

# ============================================================
#  4. Train/Test Split
# ============================================================
print("\nSplitting train/test ...")

# Time-based split (80/20)
split_idx = int(0.8 * len(agg))
train, test = agg.iloc[:split_idx], agg.iloc[split_idx:]

features = ["hour", "dayofweek", "month", "weekend", "lag1", "lag24"]
target = "count"

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# ============================================================
#  5. Model Training & Evaluation
# ============================================================
print("\nTraining models ...")

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

results = []
for name, model in models.items():
    print(f"\n▶ Training {name} ...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": mae, "R2": r2})
    print(f"{name}: MAE={mae:.3f}, R2={r2:.3f}")

# Results summary
results_df = pd.DataFrame(results).sort_values("MAE")
print("\nModel Comparison:\n", results_df)

# ============================================================
#  6. Visualization
# ============================================================
print("\nPlotting results ...")
plt.style.use("seaborn-v0_8-whitegrid")

# --- Predicted vs Actual (for best model)
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
preds = best_model.predict(X_test)

plt.figure(figsize=(10,4))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(preds[:200], label="Predicted")
plt.title(f"Actual vs Predicted Counts ({best_model_name})")
plt.xlabel("Time index")
plt.ylabel("Hourly Complaint Count")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Visualization — compare all models
# ============================================================
plt.style.use("seaborn-v0_8-whitegrid")
sample_n = min(200, len(y_test))  # show up to 200 points

for name, model in models.items():
    print(f"Plotting results for {name} ...")
    preds = model.predict(X_test)

    plt.figure(figsize=(10,4), dpi=100)
    plt.plot(y_test.values[:sample_n], label="Actual", linewidth=1.8)
    plt.plot(preds[:sample_n], label="Predicted", linewidth=1.8)
    plt.title(f"Actual vs Predicted Counts ({name})")
    plt.xlabel("Time index")
    plt.ylabel("Hourly Complaint Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"actual_vs_predicted_{name}.png")
    plt.close()

print("✅ Saved plots for all models:")
print("  actual_vs_predicted_LinearRegression.png")
print("  actual_vs_predicted_RandomForest.png")
print("  actual_vs_predicted_XGBoost.png")


# --- Feature importance (for tree models)
if best_model_name in ["RandomForest", "XGBoost"]:
    importances = best_model.feature_importances_
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
    plt.figure(figsize=(6,4))
    sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis")
    plt.title(f"Feature Importance – {best_model_name}")
    plt.tight_layout()
    plt.show()

# --- Correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(train[features + [target]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================
#  7. Save Outputs
# ============================================================
results_df.to_csv("model_results_2025.csv", index=False)
print("\nSaved results to model_results_2025.csv")
print("\nDone ✅  – Pipeline executed successfully.")
