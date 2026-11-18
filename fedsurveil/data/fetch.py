"""Download the MedMNIST v2 files (28 px) and record their checksums.

Run: python -m fedsurveil.data.fetch
IP102 is not downloaded here: it is a 3 GB Google Drive folder. See ip102.py.
"""

import hashlib
import json
import urllib.request

from fedsurveil import config

ZENODO_RECORD = "10519652"
RAW_DIR = config.DATA_DIR / "raw"
MANIFEST = config.DATA_DIR / "manifest.json"


def _url(name: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}.npz?download=1"


def _hash(path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zenodo_md5s() -> dict[str, str]:
    """md5 per file as published by Zenodo; empty if the API cannot be reached."""
    try:
        with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=30
        ) as r:
            files = json.load(r)["files"]
    except (OSError, KeyError, ValueError):
        return {}
    return {f["key"]: f["checksum"].removeprefix("md5:") for f in files}


def fetch_all() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    published = zenodo_md5s()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for name in config.MEDMNIST_DATASETS:
        path = RAW_DIR / f"{name}.npz"
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(_url(name), path)
        md5 = _hash(path, "md5")
        expected = published.get(f"{name}.npz")
        if expected and md5 != expected:
            path.unlink()
            raise RuntimeError(f"{name}: md5 {md5} does not match Zenodo ({expected})")
        manifest[name] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _hash(path, "sha256"),
            "md5": md5,
            "md5_checked_against_zenodo": bool(expected),
            "source": f"https://zenodo.org/records/{ZENODO_RECORD}",
        }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    for key, value in fetch_all().items():
        print(key, value["bytes"], value["sha256"][:12], value["md5_checked_against_zenodo"])
