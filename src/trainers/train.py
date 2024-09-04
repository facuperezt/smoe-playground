import json
from typing import Any, Dict
import torch
import tqdm
from src.data import DataLoader
from src.trainers.schedulers import LogarithmicResetLRScheduler
import wandb

__all__ = [
    "TrainWithSyntheticData",
    "TrainWithRealData"
]

class Trainer:
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._get_data = None

    def train(self, cfg: Dict[str, Any]):
        run = wandb.init(
            project=cfg.pop("project"),
            notes=cfg.pop("notes", ""),
            tags=cfg.pop("tags", ""),
            mode="online",
        )
        wandb.config = {"train_configs": cfg, "model_configs": self.model.cfg}
        optim = torch.optim.AdamW(self.model.parameters(), lr=cfg["learning_rate"])
        sched_lr = LogarithmicResetLRScheduler(optim)
        self.model.to("cuda")
        self.model.train()
        pbar = tqdm.tqdm(total=cfg["epochs"], desc=f"loss: {0.00:.5f}")
        for epoch in range(int(cfg["epochs"])):
            optim.zero_grad()
            data = self.get_data()
            out = self.model(data["input"])
            loss = self.model.loss(data["input"], out, data["loss"])
            loss["loss"].backward()
            wandb.log({"loss": loss, "learning_rate": sched_lr.get_lr()[0]})
            optim.step()
            sched_lr.step(loss["loss"].item())
            pbar.update(epoch - pbar.n)
            pbar.desc = f"loss: {loss['loss'].item():.5f}"

        wandb.log_artifact(self.model)
        return self.model

    @property
    def model(self) -> torch.nn.Module:
        return self._model
    
    def get_data(self) -> Any:
        if self._get_data is None:
            raise RuntimeError("You have to store a function in self._get_data.")
        return self._get_data()


class TrainWithSyntheticData(Trainer):
    def __init__(self, model: torch.nn.Module, num_blocks: int = 1500):
        super().__init__(model)
        self.dataloader = DataLoader("synthetic", self.model.n_kernels, self.model.block_size)  # generate synthetic data
        self._get_data = lambda: self.dataloader.get(m=num_blocks)


class TrainWithRealData(Trainer):
    def __init__(self, model: torch.nn.Module, batch_size: int = 15, img_size: int = 384):
        super().__init__(model)
        self.dataloader = DataLoader("dataset", self.model.n_kernels, self.model.block_size, "professional_photos", img_size=img_size, batch_size=batch_size)  # get real data
        self._get_data = self.dataloader.get       
