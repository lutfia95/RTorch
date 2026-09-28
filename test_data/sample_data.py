

import argparse
import os

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a large synthetic PyTorch dataset file."
    )
    parser.add_argument(
        "output",
        help="Output .pt file.",
    )
    parser.add_argument(
        "--size-gb",
        type=float,
        default=10.0,
        help="Approximate output size in GB.",
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=4096,
        help="Number of float32 values per sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bytes_per_value = torch.tensor([], dtype=torch.float32).element_size()
    bytes_per_sample = args.sample_length * bytes_per_value
    target_bytes = int(args.size_gb * 1024**3)

    number_of_samples = max(1, target_bytes // bytes_per_sample)

    print(f"Samples: {number_of_samples:,}")
    print(f"Values per sample: {args.sample_length:,}")
    print(f"Target size: {args.size_gb:.2f} GB")

    data = {
        "samples": torch.randn(
            number_of_samples,
            args.sample_length,
            dtype=torch.float32,
        ),
        "labels": torch.randint(
            low=0,
            high=10,
            size=(number_of_samples,),
            dtype=torch.int64,
        ),
    }

    torch.save(data, args.output)

    actual_size = os.path.getsize(args.output) / 1024**3
    print(f"Written: {args.output}")
    print(f"Actual size: {actual_size:.2f} GB")


if __name__ == "__main__":
    main()
