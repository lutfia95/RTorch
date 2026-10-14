import argparse
import sys
from pathlib import Path

import torch

from hdf5_benchmark import add_common_arguments, run_benchmark

# python python_loader_test.py /mnt/e/EquiLoad/mptraj/train.h5 --r-max 5.0 --batch-size 2 --repeats 5  --json python-results.json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def _make_local_equitrain_importable() -> None:
    root = Path(__file__).resolve().parent
    checkout = root / "equitrain"
    if (checkout / "equitrain" / "__init__.py").is_file():
        sys.path.insert(0, str(checkout))


def main() -> None:
    args = parse_args()
    _make_local_equitrain_importable()
    try:
        from equitrain.data.atomic import AtomicNumberTable
        from equitrain.data.backend_torch.loaders_impl import DynamicGraphLoader
        from equitrain.data.format_hdf5 import HDF5GraphDataset
    except ImportError as error:
        raise ImportError(
            f"Unable to import Equitrain's torch loader: {error}\n"
            "Install the checked-out Equitrain torch dependencies with "
            "`python -m pip install -e './equitrain[torch]'`."
        ) from error

    def build_loader(numbers: list[int]):
        dataset = HDF5GraphDataset(
            args.path,
            r_max=args.r_max,
            atomic_numbers=AtomicNumberTable(numbers),
        )
        return DynamicGraphLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
        )

    run_benchmark(
        f"Equitrain/Python / PyTorch {torch.__version__}",
        args,
        build_loader,
    )


if __name__ == "__main__":
    main()
