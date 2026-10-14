# test_download.py
# Here I test my download code WITHOUT the internet, using a tiny fake zip file.
# I run all tests with: pytest -v

import zipfile

import pytest

import laser_defect
from laser_defect.download import check_size, extract_zip


def test_package_imports():
    # If this passes, my package is installed correctly.
    assert laser_defect.__version__ == "0.1.0"


def test_extract_zip_creates_dataset_folder(tmp_path):
    # tmp_path is a temporary empty folder pytest gives me; it is deleted afterwards.
    # I build a tiny zip with the same top folder name as the real one.
    fake_zip = tmp_path / "Dataset.zip"
    with zipfile.ZipFile(fake_zip, "w") as z:
        z.writestr("Dataset/hello.txt", "test")

    out = extract_zip(fake_zip, tmp_path / "raw")

    # I check the file really came out of the zip.
    assert (out / "hello.txt").read_text() == "test"


def test_check_size_rejects_incomplete_file(tmp_path):
    # I create a 10-byte file but claim it should be 100 bytes.
    f = tmp_path / "broken.zip"
    f.write_bytes(b"0123456789")

    # My code must raise an error AND delete the broken file.
    with pytest.raises(RuntimeError):
        check_size(f, expected_bytes=100)
    assert not f.exists()
