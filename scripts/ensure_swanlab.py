"""Install and verify the pinned SDK once per node, before launching workers.

An adjacent wheel allows existing training images to upgrade offline. Otherwise
pip uses the configured index. Run with the same Python used for training.
"""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

VERSION = "0.10.5"
WHEEL = f"swanlab-{VERSION}-py3-none-any.whl"
WHEEL_SHA256 = "e91a20cac5a5d186e583886a3d349f5cd3e57ffe347ac8ecb3c992ef7e1226df"


def main():
    try:
        installed = metadata.version("swanlab")
    except metadata.PackageNotFoundError:
        installed = None
    if installed != VERSION:
        wheel = Path(__file__).with_name(WHEEL)
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
        if wheel.is_file():
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
                raise RuntimeError("SwanLab wheel checksum mismatch")
            # The existing training image already supplies the SDK dependencies.
            command += ["--no-index", "--no-deps", str(wheel)]
        else:
            command += [f"swanlab=={VERSION}"]
        subprocess.run(command, check=True)
    import swanlab
    actual = metadata.version("swanlab")
    if actual != VERSION or swanlab.__version__ != VERSION:
        raise RuntimeError(f"Expected SwanLab {VERSION}, found {actual} at {swanlab.__file__}")
    print(json.dumps({"swanlab_version": actual, "python": sys.executable,
                      "module": swanlab.__file__}), flush=True)


if __name__ == "__main__":
    main()
