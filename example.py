import datetime
import json
from typing import Any, Dict
import torch
from src.models.base_model import SmoeModel
from src.models import VariationalAutoencoder, ConvolutionalAutoencoder, ResNet, VqVae, Elvira2023Small, Elvira2023Full
from src.trainers import TrainWithSyntheticData, TrainWithRealData

def get_class_name(instance) -> str:
    return repr(instance.__class__).strip("'>").split(".")[-1]

def train_with_synth_data(model_config: Any, run_cfg: Dict[str, Any]):
    # model = VariationalAutoencoder("manual_simple_ae.json")
    # model = ResNet(n_kernels=n_kernels, block_size=32)
    model: SmoeModel = VqVae(model_config)
    # model.load_state_dict(torch.load(f"{get_class_name(model)}_synth_data"))
    trainer = TrainWithSyntheticData(model, num_blocks=1500)
    try:
        trainer.train({**run_cfg}, "disabled")
    except Exception as e:
        print(e)
    finally:
        name = str(datetime.datetime.now()).replace(" ", "__").replace(":", "_")
        path = f'{get_class_name(model)}_{name}_synth_data'
        model.save_state_dict(path)

def finetune_with_real_data(model_config: Any, run_cfg: Dict[str, Any]):
    # model: SmoeModel = VariationalAutoencoder("wide/manual_simple_ae.json")
    # model = ResNet(n_kernels=n_kernels, block_size=32)
    model: SmoeModel = VqVae(model_config)
    model.load_state_dict(torch.load(f"{get_class_name(model)}_synth_data"))
    trainer = TrainWithRealData(model, batch_size=15)
    try:
        trainer.train(run_cfg)
    except Exception as e:
        print(e)
    finally:
        name = str(datetime.datetime.now()).replace(" ", "__").replace(":", "_")
        path = f'{get_class_name(model)}_{name}_finetune_real_data'
        model.save_state_dict(path)


if __name__ == "__main__":
    with open("src/trainers/configs/simple_training.json", "r") as f:
        cfg: Dict[str, Any] = json.load(f)
    for i in range(1, 6):
        train_with_synth_data(model_config=f"base_vqgan_{i}.json", run_cfg={**cfg, "name": f"{i}_kernels_synth"})
        # finetune_with_real_data(model_config=f"base_vqgan_{i}.json", run_cfg={**cfg, "name": f"{i}_kernels_real"})