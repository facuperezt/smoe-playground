import json
from typing import Any, Dict
import torch
# from src.models.elvira import Elvira2023Small, Elvira2023Full
from src.models.facu import VariationalAutoencoder, ConvolutionalAutoencoder, ResNetWeirdness
from src.trainers import TrainWithSyntheticData, TrainWithRealData

def get_class_name(instance) -> str:
    return repr(instance.__class__).strip("'>").split(".")[-1]

def train_with_synth_data(run_cfg: Dict[str, Any]):
    # model = VariationalAutoencoder("manual_simple_ae.json")
    model = ResNetWeirdness(n_kernels=1, block_size=16)
    # model.load_state_dict(torch.load("vae_synth_data.pth"))
    trainer = TrainWithSyntheticData(model, num_blocks=1500)
    try:
        trainer.train(run_cfg)
    except Exception as e:
        print(e)
    finally:
        torch.save(model.state_dict(), f"{get_class_name(model)}_synth_data.pth")


def finetune_with_real_data(run_cfg: Dict[str, Any]):
    model = VariationalAutoencoder("wide/manual_simple_ae.json")
    model.load_state_dict(torch.load(f"{get_class_name(model)}_synth_data.pth"))
    trainer = TrainWithRealData(model)
    try:
        trainer.train(run_cfg)
    except Exception as e:
        print(e)
    finally:
        torch.save(model.state_dict(), f"{get_class_name(model)}_finetune_real_data.pth")


if __name__ == "__main__":
    with open("src/trainers/configs/simple_training.json", "r") as f:
        cfg: Dict[str, Any] = json.load(f)
    train_with_synth_data(run_cfg=cfg)
    # finetune_with_real_data()