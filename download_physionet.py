import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/"
OUTPUT_DIR = "data/physionet-2019/training_setB"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading PhysioNet file list...")

request = urllib.request.Request(
    BASE_URL,
    headers={"User-Agent": "Mozilla/5.0"}
)

with urllib.request.urlopen(request) as response:
    html = response.read().decode("utf-8")

files = sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))

print(f"Found {len(files)} patient files.")

def download(filename):
    output = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output):
        return filename, "already exists"

    url = BASE_URL + filename

    try:
        urllib.request.urlretrieve(url, output)
        return filename, "downloaded"
    except Exception as e:
        return filename, f"ERROR: {e}"

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(download, f) for f in files]

    completed = 0

    for future in as_completed(futures):
        filename, status = future.result()
        completed += 1

        if completed % 100 == 0:
            print(f"{completed}/{len(files)} completed")

print("\nDownload finished.")
print(f"Files saved to: {OUTPUT_DIR}")