import datetime
import gc
import json
import os
import tempfile
from typing import Any, Dict
import torch
from src.models.base_model import SmoeModel
from src.models import VariationalAutoencoder, ConvolutionalAutoencoder, ResNet, VqVae, Elvira2023Small, Elvira2023Full
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

def finetune_with_real_data(model_config: Any, run_cfg: Dict[str, Any], batch_size: int = 15):
    # model: SmoeModel = VariationalAutoencoder("wide/manual_simple_ae.json")
    # model = ResNet(n_kernels=n_kernels, block_size=32)
    model: SmoeModel = VqVae(model_config)
    model.load_model(torch.load(f"{get_class_name(model)}_synth_data"))
    trainer = TrainWithRealData(model, batch_size=batch_size)
    run_name = run_cfg.get("name", "")
    try:
        trainer.train(run_cfg)
    except KeyboardInterrupt as e:
        print("Interrupted training manually, going to next model :)")
    finally:
        now = str(datetime.datetime.now()).replace(" ", "__").replace(":", "_")
        if run_name != "":
            path = f"{run_name}_{now}"
        else:
            path = f'{get_class_name(model)}_{now}_real_data'
        model.save_model(path)

print("Some change")
if __name__ == "__main__":
    with open("src/trainers/configs/simple_training.json", "r") as f:
        train_config: Dict[str, Any] = json.load(f)
    for model_class in [VariationalAutoencoder, VqVae]:
        model_class: SmoeModel
        for block_size in [8, 16]:
            for n_kernels in range(1, 6):
                tmp_file_path = os.path.join(tempfile.gettempdir(), "temp_config_training_smoe_playground.json")
                with open(os.path.join(model_class._saves_path.replace(r"saves", "configs"), "base.json"), "r") as base_cfg:
                    adapted_cfg = json.load(base_cfg)
                adapted_cfg["smoe_configs"]["n_kernels"] = n_kernels
                adapted_cfg["smoe_configs"]["block_size"] = block_size
                with open(tmp_file_path, "w") as tmp_cfg:
                    json.dump(adapted_cfg, tmp_cfg)
                model = model_class(config_path=tmp_file_path)
                analyse_model_size(model)
                train_with_synth_data(model, run_cfg={**train_config, "name": f"{model.__class__.__name__}_{n_kernels}_k_{block_size}_bs_synth"}, num_blocks=500)
                # finetune_with_real_data(model_config=f"base_vqgan_{i}.json", run_cfg={**cfg, "name": f"{i}_kernels_real"})
                del model
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    os.remove(tmp_file_path)
                except FileNotFoundError as e:
                    print("WARNING: FILE SHOULD EXIST")
                    print(e)