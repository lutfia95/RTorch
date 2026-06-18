from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class TensorFolderDataset(Dataset):
    def __init__(self, root: str):
        root_path = Path(root)
        self.classes = sorted(path.name for path in root_path.iterdir() if path.is_dir())
        self.samples = []

        for label, class_name in enumerate(self.classes):
            class_dir = root_path / class_name
            for path in sorted(class_dir.rglob("*.pt")):
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        return torch.load(path), label


dataset = TensorFolderDataset("dataset")
print(dataset.classes)
print(len(dataset))

sample, label = dataset[0]
print(sample.shape, label)

loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
for data, labels in loader:
    print(data.shape, labels)

