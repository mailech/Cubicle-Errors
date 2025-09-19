
from typing import List


class ToyDataset:
    def __init__(self, data: List[int]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> int:
        return self.data[idx]
