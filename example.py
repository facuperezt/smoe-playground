import datetime
import gc
import json
import os
import tempfile
from typing import Any, Dict
import torch
from src.models.base_model import SmoeModel
from src.models import VariationalAutoencoder, ConvolutionalAutoencoder, ResNet, VqVae, Elvira2023Small, Elvira2023Full, Vgg16
from src.trainers import TrainWithSyntheticData, TrainWithRealData

def analyse_model_size(model: SmoeModel):
    params = sum(p.numel() for p in model.parameters())*1e-6
    gigs = sum(p.numel()*p.element_size() for p in model.parameters())*1e-9
    print(f"Model '{model.__class__.__name__}' has {params:.3f} million parameters.\nModel size: {gigs:.3f}GB")

def get_class_name(instance) -> str:
    return repr(instance.__class__).strip("'>").split(".")[-1]

def train_with_synth_data(model: SmoeModel, run_cfg: Dict[str, Any], num_blocks: int = 1000):
    trainer = TrainWithSyntheticData(model, num_blocks=num_blocks)
    run_name = run_cfg.get("name", "")
    try:
        trainer.train({**run_cfg}, "online")
    except KeyboardInterrupt as e:
        print("Interrupted training manually, going to next model :)")
    finally:
        now = str(datetime.datetime.now()).replace(" ", "__").replace(":", "_")
        if run_name != "":
            path = f"{run_name}_{now}"
        else:
            path = f'{get_class_name(model)}_{now}_synth_data'
        model.save_model(path)

def finetune_with_real_data(model: SmoeModel, run_cfg: Dict[str, Any], batch_size: int = 15):
    load_from = run_cfg.get("load_model_from", "")
    if load_from == "":
        load_from = f'{get_class_name(model)}_<latest>_synth_data'
    model.load_model(load_from)
    trainer = TrainWithRealData(model, batch_size=batch_size)
    run_name = run_cfg.get("name", "")
    try:
        trainer.train({**run_cfg}, "online")
    except KeyboardInterrupt as e:
        print("Interrupted training manually, going to next model :)")
    finally:
        now = str(datetime.datetime.now()).replace(" ", "__").replace(":", "_")
        if run_name != "":
            path = f"{run_name}_{now}"
        else:
            path = f'{get_class_name(model)}_{now}_real_data'
        model.save_model(path)

if __name__ == "__main__":
    with open("src/trainers/configs/simple_training.json", "r") as f:
        train_config: Dict[str, Any] = json.load(f)
    for model_class in [ResNet]:
        model_class: SmoeModel
        for block_size in [8, 16]:
            for n_kernels in range(4, 5):
                tmp_file_path = os.path.join(tempfile.gettempdir(), "temp_config_training_smoe_playground.json")
                with open(os.path.join(model_class._saves_path.replace(r"saves", "configs"), "base.json"), "r") as base_cfg:
                    adapted_cfg = json.load(base_cfg)
                adapted_cfg["smoe_configs"]["n_kernels"] = n_kernels
                adapted_cfg["smoe_configs"]["block_size"] = block_size
                with open(tmp_file_path, "w") as tmp_cfg:
                    json.dump(adapted_cfg, tmp_cfg)
                model = model_class(config_path=tmp_file_path)
                analyse_model_size(model)
                # train_with_synth_data(model, run_cfg={
                #     **train_config,
                #     "name": f"{model.__class__.__name__}_{n_kernels}_k_{block_size}_bs_synth"
                #     }, num_blocks=500)
                finetune_with_real_data(model, run_cfg={
                    **train_config,
                    "load_model_from": f"{model.__class__.__name__}_{n_kernels}_k_{block_size}_bs_real_ft_<latest>",
                    "name": f"{model.__class__.__name__}_{n_kernels}_k_{block_size}_bs_real_double_ft"
                    }, batch_size=4)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    os.remove(tmp_file_path)
                except FileNotFoundError as e:
                    print("WARNING: FILE SHOULD EXIST")
                    print(e)