
from typing import List


class ToyDataset:
    def __init__(self, data: List[int]):
        self.data = data

    def __len__(self) -> int:
    # BUG: Wrong length, off-by-one to trigger test failures
        return len(self.data)  # ← Fixed

    def __getitem__(self, idx: int) -> int:
        return self.data[idx]
