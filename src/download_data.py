"""Download NYC green taxi trips, one file per month, from the city's public data.

Usage: python src/download_data.py                    (every month)
       python src/download_data.py 2019-10 2019-11    (only these months)
"""
import sys
import urllib.request

from common import ALL_MONTHS, DATA_DIR

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

months = sys.argv[1:] or ALL_MONTHS

DATA_DIR.mkdir(exist_ok=True)
for month in months:
    file = DATA_DIR / f"green_tripdata_{month}.parquet"
    if file.exists():
        print(f"{month} already downloaded")
        continue
    request = urllib.request.Request(f"{BASE_URL}/{file.name}", headers={"User-Agent": "taxi-class-5"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    file.write_bytes(content)  # written only after the whole file has arrived
    print(f"{month} downloaded")
