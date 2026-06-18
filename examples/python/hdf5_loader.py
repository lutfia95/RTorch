import h5py
import torch
from torch.utils.data import DataLoader, Dataset


class H5Structures(Dataset):
    def __init__(self, path: str, key: str):
        self.path = path
        self.key = key
        with h5py.File(path, "r") as handle:
            self.spans = [
                (int(structure["offset"]), int(structure["length"]))
                for structure in handle["structures"]
            ]

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, index: int):
        offset, length = self.spans[index]
        with h5py.File(self.path, "r") as handle:
            values = handle[self.key][offset : offset + length]
        return torch.as_tensor(values, dtype=torch.float32), 0


dataset = H5Structures("test_data/train.h5", "positions")
sample, label = dataset[0]
print(sample.shape, label)

loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
for values, labels in loader:
    print(values.shape, labels)

