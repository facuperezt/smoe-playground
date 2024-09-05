import datetime
import json
from typing import Any, Dict
import torch
# from src.models.elvira import Elvira2023Small, Elvira2023Full
from src.models.facu import VariationalAutoencoder, ConvolutionalAutoencoder, ResNet, VqVae
from src.trainers import TrainWithSyntheticData, TrainWithRealData

def get_class_name(instance) -> str:
    return repr(instance.__class__).strip("'>").split(".")[-1]

def train_with_synth_data(model_config: Any, run_cfg: Dict[str, Any]):
    # model = VariationalAutoencoder("manual_simple_ae.json")
    # model = ResNet(n_kernels=n_kernels, block_size=32)
    model = VqVae(model_config)
    # model.load_state_dict(torch.load("vae_synth_data.pth"))
    trainer = TrainWithSyntheticData(model, num_blocks=1500)
    try:
        trainer.train({**run_cfg})
    except Exception as e:
        print(e)
    finally:
        torch.save(model.state_dict(), f"{get_class_name(model)}_{datetime.datetime.now()}_synth_data.pth")


def finetune_with_real_data(run_cfg: Dict[str, Any]):
    model = VariationalAutoencoder("wide/manual_simple_ae.json")
    model.load_state_dict(torch.load(f"{get_class_name(model)}_synth_data.pth"))
    trainer = TrainWithRealData(model)
    try:
        trainer.train(run_cfg)
    except Exception as e:
        print(e)
    finally:
        torch.save(model.state_dict(), f"{get_class_name(model)}_{datetime.datetime.now()}_finetune_real_data.pth")


if __name__ == "__main__":
    with open("src/trainers/configs/simple_training.json", "r") as f:
        cfg: Dict[str, Any] = json.load(f)
    for i in range(1, 6):
        train_with_synth_data(model_config=f"base_vqgan_{i}.json", run_cfg={**cfg, "name": f"{i}_kernels"})
    # finetune_with_real_data()