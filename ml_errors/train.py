
import torch
import torch.nn as nn
import torch.optim as optim


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)  # BUG: Should be 2 classes for CrossEntropyLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_data(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(n, 10)
    # Two classes 0/1
    y = (torch.rand(n) > 0.5).long()
    return x, y


def train_one_epoch() -> float:
    model = TinyNet()
    x, y = make_data()
    opt = optim.SGD(model.parameters(), lr=0.1)
    # BUG: Using CrossEntropyLoss with 1-dim output will raise or behave incorrectly
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(5):
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)  # Intentional shape mismatch for failure
        loss.backward()
        opt.step()
    # Return dummy accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc


if __name__ == "__main__":
    print(train_one_epoch())
