import argparse
import sys

import h5py


def print_h5_item(name: str, obj: h5py.Dataset | h5py.Group) -> None:
    if isinstance(obj, h5py.Dataset):
        print(f"{name}")
        print(f"  type: dataset")
        print(f"  shape: {obj.shape}")
        print(f"  dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{name}")
        print(f"  type: group")


def check_all(path: str) ->  None:
    
    with h5py.File(path, "r") as f:
        MAGIC = f["MAGIC"]
        structures = f["structures"]
        atomic_numbers = f["atomic_numbers"]
        forces = f["forces"]
        positions = f["positions"]
        
        print(f'File MAGIC: \n {MAGIC.keys()}')
        print(f["MAGIC"]["MAGIC_STRING"][()])
        
        print(f'File atomic_numbers')
        print("Dataset object:")
        print(atomic_numbers)

        print("\nShape:")
        print(atomic_numbers.shape)

        print("\nDtype:")
        print(atomic_numbers.dtype)

        print("\nAll atomic numbers:")
        print(atomic_numbers[:])
        
        print(f'File forces')
        print("Dataset object:")
        print(forces)

        print("\nShape:")
        print(forces.shape)

        print("\nDtype:")
        print(forces.dtype)

        print("\nAll forces:")
        print(forces[:])
        
        print(f'File positions')
        print("Dataset object:")
        print(positions)

        print("\nShape:")
        print(positions.shape)

        print("\nDtype:")
        print(positions.dtype)

        print("\nAll positions:")
        print(positions[:])
        
        print(f'File structures-offet: \n {structures["offset"][:]}')
        print(f'File structures-length: \n {structures["length"][:]}')
        print(f'File structures-cell: \n {structures["cell"][:]}')
        print(f'File structures-pbc: \n {structures["pbc"][:]}')
        print(f'File structures-energy: \n {structures["energy"][:]}')
        print(f'File structures-stress: \n {structures["stress"][:]}')
        print(f'File structures-virials: \n {structures["virials"][:]}')
        print(f'File structures-dipole: \n {structures["dipole"][:]}')
        print(f'File structures-energy_weight: \n {structures["energy_weight"][:]}')
        print(f'File structures-forces_weight: \n {structures["forces_weight"][:]}')
        print(f'File structures-stress_weight: \n {structures["stress_weight"][:]}')
        print(f'File structures-virials_weight: \n {structures["virials_weight"][:]}')
        print(f'File structures-dipole_weight: \n {structures["dipole_weight"][:]}')
  

def inspect_h5(path: str) -> None:
    print(f"\nFile: {path}")

    with h5py.File(path, "r") as f:
        print("\nTop-level keys:")
        for key in f.keys():
            print(f"  {key}")

        print("\nFull structure:")
        f.visititems(print_h5_item)
        
        


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="HDF5 files to inspect")
    args = parser.parse_args()

    for path in args.paths:
        inspect_h5(path)
        check_all(path)

    return 0


if __name__ == "__main__":
    sys.exit(main())