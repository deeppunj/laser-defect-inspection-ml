# download.py
# Here I download the laser beam welding dataset from Figshare, check that the
# download is complete, and unzip it into data/raw/.
#
# Dataset: Walther et al. (2025), "Dataset for weld seam analysis and discontinuity
# prediction in laser beam welding scenarios", Data in Brief.
# DOI: https://doi.org/10.6084/m9.figshare.27877413.v1  (licence: CC BY 4.0)
#
# I run it from the project folder with:
#     python -m laser_defect.download

import zipfile                 # I use zipfile to unpack the downloaded .zip
from pathlib import Path       # I use Path to build file paths that work on Mac and Windows

import requests                # I use requests to download the file from the internet

# The direct download link for Dataset.zip on Figshare.
DATA_URL = "https://ndownloader.figshare.com/files/50684688"

# The exact size of the file in bytes, according to the Figshare API.
# I compare against this number to be sure the download is not cut off.
EXPECTED_BYTES = 90_442_191

# Where I keep the data. Path("data") means "a folder called data, inside
# the folder I run the command from" (my project folder).
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "Dataset.zip"     # the "/" joins folder and file names
RAW_DIR = DATA_DIR / "raw"


def download_zip(url=DATA_URL, target=ZIP_PATH, expected_bytes=EXPECTED_BYTES):
    # I skip the download if a complete copy already exists, to save time.
    if target.exists() and target.stat().st_size == expected_bytes:
        print(f"Already downloaded: {target}")
        return target

    # I create the data/ folder if it does not exist yet.
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset (about {expected_bytes / 1e6:.0f} MB)...")
    # stream=True means I download in pieces instead of loading 90 MB into memory at once.
    with requests.get(url, stream=True, timeout=60) as response:
        # If the server answers with an error (e.g. 404 Not Found), I stop here
        # with a clear message instead of saving an error page as my "dataset".
        response.raise_for_status()
        with open(target, "wb") as f:          # "wb" = write bytes (binary file)
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB pieces
                f.write(chunk)

    check_size(target, expected_bytes)
    print(f"Saved {target}")
    return target


def check_size(path, expected_bytes):
    # I compare the real file size with the expected size.
    # If they differ, the download was interrupted, so I delete the broken file
    # and stop with an error, so I never work with half a dataset.
    actual = path.stat().st_size
    if actual != expected_bytes:
        path.unlink()                          # unlink = delete the file
        raise RuntimeError(
            f"Download incomplete: got {actual} bytes, expected {expected_bytes}. "
            "I deleted the broken file; please run the command again."
        )


def extract_zip(zip_path=ZIP_PATH, out_dir=RAW_DIR):
    # I unzip the archive into data/raw/. If it is already unzipped, I skip it.
    marker = out_dir / "Dataset"               # the zip contains one top folder: Dataset/
    if marker.exists():
        print(f"Already extracted: {marker}")
        return marker

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print(f"Extracted to {marker}")
    return marker


def summarise(dataset_dir):
    # I count what I downloaded, as a quick sanity check.
    labels = sorted((dataset_dir / "Preprocessed" / "LWIR" / "Labels").glob("*.csv"))
    probes = sorted((dataset_dir / "Raw" / "Inductive_Probes").rglob("*.csv"))
    print(f"Label files (one per weld with thermal video): {len(labels)}")
    print(f"Raw probe files (one per weld):               {len(probes)}")
    return len(labels), len(probes)


# This block only runs when I start this file directly with
# "python -m laser_defect.download", not when another file imports it.
if __name__ == "__main__":
    download_zip()
    dataset_dir = extract_zip()
    summarise(dataset_dir)
