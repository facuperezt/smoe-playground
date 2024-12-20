from typing import Any, Dict, Literal, Optional, Tuple
import torch
import tqdm
from src.data import DataLoader
from src.models.base_model import SmoeModel
from src.trainers.schedulers import LogarithmicResetLRScheduler
from src.utils import Img2Block
import wandb

from src.utils import Block2Img

__all__ = [
    "TrainWithSyntheticData",
    "TrainWithRealData"
]

class Trainer:
    def __init__(self, model: SmoeModel):
        self._model = model
        self._get_training_data = None
        self._get_eval_data = None

    def train(self, cfg: Dict[str, Any], wandb_mode: str = "disabled"):
        self.run = wandb.init(
            project=cfg.pop("project"),
            notes=cfg.pop("notes", ""),
            tags=cfg.pop("tags", ""),
            name=cfg.pop("name", None),
            mode=wandb_mode,
        )
        wandb.config = {"train_configs": cfg, "model_configs": self.model.cfg}
        optim = torch.optim.AdamW(self.model.parameters(), lr=cfg["learning_rate"])
        sched_lr = LogarithmicResetLRScheduler(optim, **cfg["scheduler_configs"], callbacks_on_reset=[self.log_images, self.model.save_model])
        self.model.to("cuda")
        self.model.train()
        self.eval_input = self.get_data("eval").cuda().requires_grad_(False)
        self.best_eval_recon, best_eval_loss = self.eval_model(self.eval_input)
        pbar = tqdm.tqdm(total=cfg["epochs"], desc=f"train_loss: {0.00:.5f} - eval_loss: {0.0:.5f}")
        for epoch in range(int(cfg["epochs"])):
            optim.zero_grad()
            data = self.get_data()
            out = self.model(data["input"])
            loss, logging = self.model.loss(data["input"], out, data["loss"])
            loss.backward()
            eval_recon, eval_loss = self.eval_model(self.eval_input) 
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.best_eval_recon = eval_recon
            wandb.log({"loss": loss.item(), **logging, "learning_rate": sched_lr.get_lr()[0], "Eval Reconstruction Loss": eval_loss})
            optim.step()
            sched_lr.step(eval_loss)
            pbar.update(epoch - pbar.n + 1)
            pbar.desc = f"train_loss: {loss.item():.5f} - eval_loss: {eval_loss:.5f}"

        self.run.finish()
        return self.model

    def log_images(self, eval_input: Optional[torch.Tensor] = None, best_eval_recon: Optional[torch.Tensor] = None) -> None:
        try:
            if eval_input is None:
                eval_input = self.eval_input
            if best_eval_recon is None:
                best_eval_recon =  self.best_eval_recon
            if wandb.run is not None: # and not isinstance(wandb.run, wandb.sdk.lib.disabled.RunDisabled):
                shape = Block2Img.__bases__[0]._last_unfolded_shape.clone()
                Block2Img.__bases__[0]._last_unfolded_shape[0] = 1
                wandb.log({"Eval Image": wandb.Image(self.blocks2img(eval_input).squeeze().cpu().numpy()),
                        "Eval Reconstruction": wandb.Image(self.blocks2img(best_eval_recon).squeeze().numpy())})
                Block2Img.__bases__[0]._last_unfolded_shape = shape
        except Exception as e:
            # TODO: Investigate the kind of errors this could throw (not defined for the self methods and if the tensors have wrong shape probs)
            print("Logging Image failed for SOME reason. Catching the error and printing it to avoid stuff crashing. BAD PRACTICE!")
            print("\n"+"="*len(str(e))+"\n", e, "\n"+"="*len(str(e))+"\n")


    def eval_model(self, eval_input: torch.Tensor) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            eval_recon = self.model.reconstruct_input(eval_input)
            eval_loss = torch.nn.functional.mse_loss(eval_recon, eval_input)
        return eval_recon.cpu(), eval_loss.item()

    @property
    def model(self) -> SmoeModel:
        return self._model
    
    def get_data(self, mode: Literal["train", "eval"] = "train") -> torch.Tensor:
        if mode == "train":
            if self._get_training_data is None:
                raise RuntimeError("You have to store a function in self._get_training_data.")
            return self._get_training_data()
        elif mode == "eval":
            if self._get_eval_data is None:
                raise RuntimeError("You have to store a function in self._get_eval_data.")
            return self._get_eval_data()


class TrainWithSyntheticData(Trainer):
    def __init__(self, model: torch.nn.Module, num_blocks: int = 1500, img_size: int = 384, rescale_range: Tuple[int, int] = (0, 1)):
        super().__init__(model)
        self.dataloader = DataLoader("synthetic", self.model.n_kernels, self.model.block_size, "professional_photos", img_size, rescale_range=rescale_range)  # generate synthetic data, needs a dataset path to know which validation pic to use
        self.img2blocks = Img2Block(self.model.block_size, img_size)
        self.blocks2img = Block2Img(self.model.block_size, img_size)
        self._get_training_data = lambda: self.dataloader.get(m=num_blocks, gaussian_sampling_steering_kernels=True, vmin=-7, vmax=7)
        self._get_eval_data = lambda: self.img2blocks(self.dataloader.get_valid_pic())

class TrainWithRealData(Trainer):
    def __init__(self, model: torch.nn.Module, batch_size: int = 15, img_size: int = 384, rescale_range: Tuple[int, int] = (0, 1)):
        super().__init__(model)
        self.dataloader = DataLoader("dataset", self.model.n_kernels, self.model.block_size, "professional_photos", img_size=img_size, batch_size=batch_size, rescale_range=rescale_range)  # get real data
        self.img2blocks = Img2Block(self.model.block_size, img_size)
        self.blocks2img = Block2Img(self.model.block_size, img_size)
        self._get_training_data = self.dataloader.get       
        self._get_eval_data = lambda: self.img2blocks(self.dataloader.get_valid_pic())
