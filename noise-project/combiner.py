import glob
import pandas as pd

files = glob.glob("nyc311_noise_*.csv")
df_all = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
df_all.to_csv("nyc311_noise_all_years.csv", index=False)
