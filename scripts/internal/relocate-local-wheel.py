#!/usr/bin/env python3
"""Make an EquiLoad local-compatibility wheel independent of its build path."""
from __future__ import annotations

import base64
import csv
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def main(wheel: Path) -> None:
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SystemExit(f"Expected an EquiLoad wheel, got: {wheel}")
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        raise SystemExit("patchelf is required to relocate a Linux wheel")

    with tempfile.TemporaryDirectory(prefix="equiload-wheel-") as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(directory)
        extensions = list((directory / "equiload").glob("equiload*.so"))
        records = list(directory.glob("equiload-*.dist-info/RECORD"))
        if len(extensions) != 1 or len(records) != 1:
            raise SystemExit("Wheel does not have the expected EquiLoad extension and RECORD")

        extension = extensions[0]
        subprocess.run(
            [patchelf, "--set-rpath", "$ORIGIN/../torch/lib", extension], check=True
        )
        record = records[0]
        rows = list(csv.reader(record.open(newline="")))
        extension_key = extension.relative_to(directory).as_posix()
        for row in rows:
            if row[0] == extension_key:
                row[1] = record_hash(extension)
                row[2] = str(extension.stat().st_size)
                break
        else:
            raise SystemExit("Extension is missing from wheel RECORD")
        with record.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)

        rebuilt = wheel.with_suffix(".rebuilt")
        with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(directory))
        rebuilt.replace(wheel)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: relocate-local-wheel.py PATH_TO_WHEEL")
    main(Path(sys.argv[1]))
