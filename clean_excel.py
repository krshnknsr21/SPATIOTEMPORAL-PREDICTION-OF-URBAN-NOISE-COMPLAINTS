import pandas as pd

# Columns to keep (ONLY the ones your model uses)
KEEP_COLS = [
    "created_date",
    "borough",
    "incident_zip",
    "complaint_type",
    "latitude",
    "longitude"
]

def clean_large_csv(input_path, output_path):
    chunksize = 500_000  # adjust if needed
    reader = pd.read_csv(input_path, chunksize=chunksize, low_memory=False)

    # Write header first
    first_chunk = True

    for chunk in reader:
        # Normalize column names
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        # Keep only required columns
        chunk = chunk[[c for c in KEEP_COLS if c in chunk.columns]]

        # Append to output CSV
        chunk.to_csv(output_path, mode='a', index=False, header=first_chunk)
        first_chunk = False

    print(f"Cleaned file written to: {output_path}")


# ---------------- RUN THIS ----------------
clean_large_csv("nyc311_noise_2021_2022.csv", "cleaned_2021_2022.csv")