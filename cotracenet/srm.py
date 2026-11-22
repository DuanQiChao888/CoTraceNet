"""Get or validate the public SRM kernel bank used by the SiaStegNet baseline.

This file contains our loader. The authors of ShuffleBANet did not specify a
download for their own kernel file. This pinned upstream file is a documented
choice for the reconstruction. Its values are used without further scaling.
"""

import argparse
import hashlib
import io
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np


SRM_URL = (
    "https://raw.githubusercontent.com/SiaStg/SiaStegNet/"
    "592d9e13f39f287d9c675f1b89272bcfb6a9627c/src/models/SRM_Kernels.npy"
)
SRM_SHA256 = "c53ad3a849fa64db1244152a3030c80a4d3b11fec0af81b4196d8e6dfb3db4b9"
SRM_SIZE = 3080


def _decode(data):
    if len(data) != SRM_SIZE or hashlib.sha256(data).hexdigest() != SRM_SHA256:
        raise ValueError("SRM file does not match the pinned upstream SHA-256.")
    weights = np.load(io.BytesIO(data), allow_pickle=False)
    if weights.shape != (30, 1, 5, 5) or weights.dtype != np.dtype("<f4"):
        raise ValueError("Expected 30 grayscale float32 filters of size 5 x 5.")
    return weights.astype(np.float32, copy=False)


def load_srm_kernels(path):
    """Read the pinned bank as a (30, 1, 5, 5) NumPy array; works offline."""
    return _decode(Path(path).read_bytes())


def prepare(path):
    """Verify an existing file or download the missing file once."""
    path = Path(path)
    if path.exists():
        load_srm_kernels(path)
        return
    with urlopen(SRM_URL, timeout=30) as response:
        data = response.read(SRM_SIZE + 1)
    _decode(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or verify the reference SRM bank.")
    parser.add_argument("path", nargs="?", default="SRM_Kernels.npy")
    target = parser.parse_args().path
    prepare(target)
    print(f"Verified {target}  sha256={SRM_SHA256}")
