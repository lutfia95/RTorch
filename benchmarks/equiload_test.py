import argparse
import ctypes
import os
from pathlib import Path

import torch

# python equiload_test.py /mnt/e/EquiLoad/mptraj/train.h5 --r-max 5.0 --batch-size 2 --repeats 5 --json equiload-results.json
from hdf5_benchmark import add_common_arguments, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def _preload_local_hdf5() -> None:
    """Preload a usable HDF5 runtime before importing the EquiLoad extension."""
    library_dirs = []
    if hdf5_root := os.environ.get("HDF5_ROOT"):
        library_dirs.append(Path(hdf5_root).expanduser() / "lib")
    library_dirs.append(Path(__file__).resolve().parents[1] / "third_party" / "hdf5" / "lib")

    load_errors = []
    for library_dir in library_dirs:
        candidates = (
            library_dir / "libhdf5.so.310",
            library_dir / "libhdf5.so",
            library_dir / "libhdf5.dylib",
        )
        for library in candidates:
            if not library.is_file() or library.stat().st_size == 0:
                continue
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError as error:
                load_errors.append(f"{library}: {error}")
                continue
            return

    if load_errors:
        raise OSError("Unable to preload an HDF5 runtime:\n  " + "\n  ".join(load_errors))


def main() -> None:
    args = parse_args()
    _preload_local_hdf5()
    try:
        import equiload
    except ImportError as error:
        raise ImportError(
            f"Unable to import equiload: {error}\n"
            "EquiLoad and Python's PyTorch must use the same LibTorch version and "
            "C++ ABI. Rebuild EquiLoad against the active torch package; see the "
            "README benchmark section."
        ) from error

    def build_loader(numbers: list[int]):
        return equiload.HDF5GraphLoader.from_hdf5_file(
            args.path,
            r_max=args.r_max,
            atomic_numbers=numbers,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

    run_benchmark(
        f"EquiLoad {getattr(equiload, '__version__', 'unknown')} / PyTorch {torch.__version__}",
        args,
        build_loader,
    )


if __name__ == "__main__":
    main()
