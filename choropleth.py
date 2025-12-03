import pandas as pd
import geopandas as gpd
import plotly.express as px
import numpy as np

# ==========================================
# 1. Load predicted DAILY noise per ZIP
# ==========================================

daily_zip = pd.read_csv("predictions_2026_zip_daily.csv", parse_dates=["date"])

# Ensure ZIP is 5-digit string
daily_zip["zip"] = daily_zip["zip"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)

# Convert to WEEK format (e.g., 2026-03-09/2026-03-15)
daily_zip["week"] = daily_zip["date"].dt.to_period("W").astype(str)

# Weekly mean noise per ZIP
weekly_zip = (
    daily_zip.groupby(["zip", "week"])["daily_total_noise"]
    .mean()
    .reset_index()
    .rename(columns={"daily_total_noise": "avg_daily_noise"})
)

print("Weekly ZIP prediction sample:")
print(weekly_zip.head())


# ==========================================
# 2. Load NYC ZIP population dataset
# (You created this file earlier)
# ==========================================

pop = pd.read_csv("nyc_zip_population.csv")
pop["zip"] = pop["zip"].astype(str).str.zfill(5)

# Merge population into weekly noise data
weekly_zip = weekly_zip.merge(pop, on="zip", how="left")


# ==========================================
# 3. Compute NOISE PER 1,000 residents
# ==========================================

weekly_zip["noise_per_1000"] = (weekly_zip["avg_daily_noise"] / weekly_zip["population"]) * 1000


# ==========================================
# 4. Log scaling to boost contrast
# ==========================================

weekly_zip["noise_log"] = weekly_zip["noise_per_1000"].apply(lambda x: np.log1p(x))

# Scale 0–100 range for clean visualization
mn = weekly_zip["noise_log"].min()
mx = weekly_zip["noise_log"].max()

weekly_zip["scaled_noise"] = (weekly_zip["noise_log"] - mn) / (mx - mn) * 100


# ==========================================
# 5. Load GeoJSON & prepare ZIP polygons
# ==========================================

zip_geo = gpd.read_file("nyc-zip-code-tabulation-areas-polygons.geojson")

# Your GeoJSON field is "postalCode"
zip_geo["postalCode"] = zip_geo["postalCode"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)

print("GeoJSON fields:", zip_geo.columns)


# ==========================================
# 6. Build OFFLINE choropleth (NO MAPBOX)
# ==========================================

fig = px.choropleth(
    weekly_zip,
    geojson=zip_geo.__geo_interface__,
    locations="zip",                           # ZIP in your data
    featureidkey="properties.postalCode",      # ZIP in the GeoJSON
    color="scaled_noise",                      # the scaled (log) noise density
    color_continuous_scale="Reds",             # red gradient
    range_color=(0, 100),
    animation_frame="week",                    # weekly slider
    labels={"scaled_noise": "Noise per 1000 residents (log-scaled)"},
    projection="mercator",
)

# Fit map to NYC boundaries
fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(
    title="Predicted Weekly Noise Exposure per 1,000 Residents (2026, Log Scaled)",
    margin={"r":0, "t":40, "l":0, "b":0},
)

fig.show()

# Optional: save for presentation
fig.write_html("noise_map_2026_weekly_percapita.html")