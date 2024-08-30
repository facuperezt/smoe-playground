import torch
import tqdm
from src.data import DataLoader
from src.trainers.schedulers import LogarithmicResetLRScheduler

__all__ = [
    "TrainWithSyntheticData",
    "TrainWithRealData"
]

class TrainWithSyntheticData:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.dataloader = DataLoader("synthetic", self.model.n_kernels, self.model.block_size)  # generate synthetic data
 
    def train(self, epochs: int = 10, lr: float = 1e-5):
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sched_lr = LogarithmicResetLRScheduler(optim)
        self.model.to("cuda")
        self.model.train()
        pbar = tqdm.tqdm(total=epochs, desc=f"loss: {0.00:.5f}")
        for epoch in range(epochs):
            optim.zero_grad()
            data = self.dataloader.get(m=15000)  # m blocks
            out = self.model(data["input"])
            loss = self.model.loss(data["input"], out, data["loss"])
            loss["loss"].backward()
            optim.step()
            sched_lr.step(loss["loss"].item())
            pbar.update(epoch - pbar.n)
            pbar.desc = f"loss: {loss['loss'].item():.5f}"

        return self.model


class TrainWithRealData:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.dataloader = DataLoader("dataset", self.model.n_kernels, self.model.block_size, "professional_photos", img_size=384, batch_size=25)  # get real data
 
    def train(self, epochs: int = 10, lr: float = 1e-5):
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sched_lr = LogarithmicResetLRScheduler(optim)
        self.model.to("cuda")
        self.model.train()
        pbar = tqdm.tqdm(total=epochs, desc=f"loss: {0.00:.5f}")
        for epoch in range(epochs):
            optim.zero_grad()
            data = self.dataloader.get()  # m blocks
            out = self.model(data["input"])
            loss = self.model.loss(data["input"], out, data["loss"])
            loss["loss"].backward()
            optim.step()
            sched_lr.step(loss["loss"].item())
            pbar.update(epoch - pbar.n)
            pbar.desc = f"loss: {loss['loss'].item():.5f}"

        return self.model
        
