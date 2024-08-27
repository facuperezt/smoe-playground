import torch
import tqdm
from src.data import DataLoader

__all__ = [
    "TrainWithDataloader",
]

class TrainWithDataloader:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.dataloader = DataLoader("synthetic", self.model.n_kernels, self.model.block_size)  # generate synthetic data

    def train(self, epochs: int = 10):
        optim = torch.optim.AdamW(self.model.parameters())

        self.model.to("cuda")
        self.model.train()
        for epoch in tqdm.tqdm(range(epochs)):
            optim.zero_grad()
            data = self.dataloader.get(m=100)  # m blocks
            out = self.model(data)
            loss = self.model.loss(data, out)
            loss.backward()
            optim.step()

        return self.model

        
