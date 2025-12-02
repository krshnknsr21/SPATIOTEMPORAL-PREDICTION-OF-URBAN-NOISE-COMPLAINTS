import os
import time
import csv
import requests
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

# -----------------------------
# CONFIG — tweak as you like
# -----------------------------
DATASET_ID = "erm2-nwe9"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"

# Common, high-signal fields. Add/remove as needed.
SELECT_FIELDS = [
    "unique_key", "created_date", "closed_date", "agency", "agency_name",
    "complaint_type", "descriptor", "incident_zip", "incident_address",
    "city", "borough", "latitude", "longitude", "location"
]

# Filter: include any complaint_type that starts with "Noise"
# PLUS a few historically common noise categories that don't start with "Noise".
EXTRA_NOISE_TYPES = [
    "Loud Music/Party",
    "Collection Truck Noise",
    "Car/Truck Horn",
    "Banging/Pounding"
]

# Pagination / rate limits
PAGE_LIMIT = 50000               # Socrata max is 50k
SLEEP_SECS_BETWEEN_PAGES = 0.5   # be kind to the API

# Output naming
OUT_PREFIX = "nyc311_noise_"

# Optional app token (recommended)
APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "").strip()

# -----------------------------
# Helpers
# -----------------------------
def iso(dt: datetime) -> str:
    """NYC Open Data expects ISO8601 with timezone omitted; seconds are fine."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def make_where_clause(start_dt: datetime, end_dt: datetime) -> str:
    """
    Build a SoQL WHERE that:
      - bounds by created_date
      - filters complaint_type to noise-related
    """
    # complaint_type LIKE 'Noise%' OR complaint_type IN (<extras>)
    like_noise = "complaint_type like 'Noise%%'"
    extras = " OR ".join([f"complaint_type = '{t}'" for t in EXTRA_NOISE_TYPES])
    noise_filter = f"({like_noise}" + (f" OR {extras}" if extras else "") + ")"

    time_filter = f"created_date between '{iso(start_dt)}' and '{iso(end_dt)}'"
    return f"{time_filter} AND {noise_filter}"

def fetch_window(start_dt: datetime, end_dt: datetime, session: requests.Session, out_path: str):
    """Stream a 2-year window to CSV in pages."""
    params_base = {
        "$select": ", ".join(SELECT_FIELDS),
        "$order": "created_date ASC",
        "$limit": PAGE_LIMIT,
        "$where": make_where_clause(start_dt, end_dt)
    }

    headers = {}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    offset = 0
    total_rows = 0
    wrote_header = False

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = None

        while True:
            params = dict(params_base)
            params["$offset"] = offset

            resp = session.get(BASE_URL, params=params, headers=headers, timeout=120)
            resp.raise_for_status()
            rows = resp.json()

            if not rows:
                break

            # Initialize writer with consistent fieldnames on first page
            if writer is None:
                fieldnames = SELECT_FIELDS  # fixed set from $select
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                wrote_header = True

            writer.writerows(rows)
            page_count = len(rows)
            total_rows += page_count
            offset += PAGE_LIMIT

            # polite pacing
            time.sleep(SLEEP_SECS_BETWEEN_PAGES)

    print(f"Wrote {total_rows:,} rows → {out_path}" + ("" if wrote_header else " (no header)"))

def windows_two_years(start_year=2015):
    """Yield (start_dt, end_dt, label) for 2-year windows up to today."""
    today = datetime.combine(date.today(), datetime.min.time())
    cur = datetime(start_year, 1, 1)
    while cur <= today:
        end = min(cur + relativedelta(years=2) - timedelta(seconds=1), today)
        # label like 2015_2016, etc.; last window ends at current date
        label = f"{cur.year}_{(cur + relativedelta(years=1)).year}" if end.year > cur.year else f"{cur.year}"
        yield cur, end, label
        cur = cur + relativedelta(years=2)

def main():
    session = requests.Session()
    # Helpful User-Agent
    session.headers.update({"User-Agent": "ASU-Noise-311-Downloader/1.0"})

    for start_dt, end_dt, label in windows_two_years(start_year=2015):
        # e.g., nyc311_noise_2015_2016.csv
        out_name = f"{OUT_PREFIX}{label}.csv"
        print(f"Fetching {label}: {start_dt.date()} → {end_dt.date()} ...")
        fetch_window(start_dt, end_dt, session, out_name)

    print("All done.")

if __name__ == "__main__":
    main()
