import datetime
import gc
import json
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, Tuple, Union
import torch
from src.models.base_model import SmoeModel
from src.models import VariationalAutoencoder, ConvolutionalAutoencoder, ResNet, VqVae, Elvira2023Small, Elvira2023Full, Vgg16
from src.trainers import TrainWithSyntheticData, TrainWithRealData

def make_forward_rescaling_hook(original_data_range: Tuple[int, int], rescale_data_range: Tuple[int, int], pre_hook: bool) -> Callable:
    min_value_original, max_value_original = original_data_range
    min_value_rescaled, max_value_rescaled = rescale_data_range
    assert max_value_original > min_value_original, "Data Range for pre-forward-hook is not valid."
    assert max_value_rescaled > min_value_rescaled, "Data Range for pre-forward-hook is not valid."
    len_rescaled_scale = max_value_rescaled - min_value_rescaled
    len_original_scale = max_value_original - min_value_original
    rescale_factor = len_rescaled_scale/len_original_scale
    if pre_hook:
        def _rescaling_forward_pre_hook(module: torch.nn.Module, input_tensor: Tuple[torch.Tensor]) -> torch.Tensor:
            rescaled_input_tensor = (input_tensor[0]*rescale_factor) + min_value_rescaled
            return rescaled_input_tensor
        return _rescaling_forward_pre_hook
    def _rescaling_forward_hook(module: torch.nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
        rescaled_output_tensor = (output_tensor - min_value_rescaled)/rescale_factor
        return rescaled_output_tensor 
    return _rescaling_forward_hook

def analyse_model_size(model: SmoeModel, verbose: bool = True):
    params = sum(p.numel() for p in model.parameters())*1e-6
    gigs = sum(p.numel()*p.element_size() for p in model.parameters())*1e-9
    if verbose:
        print(f"Model '{model.__class__.__name__}' has {params:.3f} million parameters.\nModel size: {gigs:.3f}GB")
    return gigs

def get_class_name(instance) -> str:
    return repr(instance.__class__).strip("'>").split(".")[-1]

def train_with_synth_data(model: SmoeModel, run_cfg: Dict[str, Any], num_blocks: Union[int, str] = 1000, rescale_data_range: Tuple[int, int] = (0, 1)):
    if isinstance(num_blocks, str):
        assert num_blocks.endswith("GB"), "num_blocks needs to be an amount of GB of VRAM if its not an int"
        vram = float(num_blocks.strip("GB"))
        model_mem = analyse_model_size(model, verbose=False) * 4  # Approx needed for Adam backprop
        rest_vram = vram - model_mem
        mem_per_block = model.block_size**2 * torch.float32.itemsize * 1e-6
        can_fit_blocks = rest_vram/3.5//mem_per_block
        num_blocks = int(can_fit_blocks)
    trainer = TrainWithSyntheticData(model, num_blocks=num_blocks, rescale_range=rescale_data_range)
    run_name = run_cfg.get("name", "")
    try:
        trainer.train({**run_cfg}, wandb_mode=wandb_mode)
    except KeyboardInterrupt as e:
        print("Interrupted training manually, going to next model :)")
    finally:
        now = str(datetime.datetime.now()).replace(" ", "__").replace(":", "_")
        trainer.log_images()
        if run_name != "":
            path = f"{run_name}_{now}"
        else:
            path = f'{get_class_name(model)}_{now}_synth_data'
        model.save_model(path)
        current_run_path = os.path.join(model.saves_path, "current_run")
        if os.path.isdir(current_run_path):
            for checkpoint in os.listdir(current_run_path):
                shutil.move(src=os.path.join(current_run_path, checkpoint), dst=os.path.join(model.saves_path, path, checkpoint))
            shutil.rmtree(current_run_path)

def finetune_with_real_data(model: SmoeModel, run_cfg: Dict[str, Any], batch_size: Union[int, str] = 15, rescale_data_range: Tuple[int, int] = (0, 1)):
    if isinstance(batch_size, str):
        assert batch_size.endswith("GB"), "num_blocks needs to be an amount of GB of VRAM if its not an int"
        vram = float(batch_size.strip("GB"))
        model_mem = analyse_model_size(model, verbose=False) * 4  # Approx needed for Adam backprop
        rest_vram = vram - model_mem
        mem_per_block = 384**2 * torch.float32.itemsize * 1e-6  # 384 is img size, hardcoded for now cause why not
        can_fit_blocks = rest_vram/3//mem_per_block
        batch_size = int(can_fit_blocks)
    load_from = run_cfg.get("load_model_from", "")
    if load_from != "":
        # load_from = f'{get_class_name(model)}_<latest>_synth_data'
        model.load_model(load_from)
    trainer = TrainWithRealData(model, batch_size=batch_size, rescale_range=rescale_data_range)
    run_name = run_cfg.get("name", "")
    try:
        trainer.train({**run_cfg}, wandb_mode=wandb_mode)
    except KeyboardInterrupt as e:
        print("Interrupted training manually, going to next model :)")
    finally:
        now = str(datetime.datetime.now()).replace(" ", "__").replace(":", "_")
        trainer.log_images()
        if run_name != "":
            path = f"{run_name}_{now}"
        else:
            path = f'{get_class_name(model)}_{now}_real_data'
        model.save_model(path)
        current_run_path = os.path.join(model.saves_path, "current_run")
        if os.path.isdir(current_run_path):
            for checkpoint in os.listdir(current_run_path):
                shutil.move(src=os.path.join(current_run_path, checkpoint), dst=os.path.join(model.saves_path, path, checkpoint))
            shutil.rmtree(current_run_path)

if __name__ == "__main__":
    wandb_mode = "online"
    with open("src/trainers/configs/simple_training_local.json", "r") as f:
        train_config: Dict[str, Any] = json.load(f)
    for model_class in [ResNet]:
        model_class: SmoeModel
        for block_size in [64]:
            for n_kernels in range(2, 5):
                tmp_file_path = os.path.join(tempfile.gettempdir(), "temp_config_training_smoe_playground.json")
                with open(os.path.join(model_class._saves_path.replace(r"saves", "configs"), "base.json"), "r") as base_cfg:
                    adapted_cfg = json.load(base_cfg)
                adapted_cfg["smoe_configs"]["n_kernels"] = n_kernels
                adapted_cfg["smoe_configs"]["block_size"] = block_size
                with open(tmp_file_path, "w") as tmp_cfg:
                    json.dump(adapted_cfg, tmp_cfg)
                model = model_class(config_path=tmp_file_path)
                model: torch.nn.Module
                if tuple(train_config.get("rescale_data_range", None)) != tuple(train_config.get("original_data_range", None)):
                    # Hooks to allow the model to work with more stable values like -1 -> 1 instead of necessarily 0 -> 1
                    model.register_forward_pre_hook(make_forward_rescaling_hook(train_config["original_data_range"], train_config["rescale_data_range"], pre_hook=True))
                    model.register_forward_hook(make_forward_rescaling_hook(train_config["original_data_range"], train_config["rescale_data_range"], pre_hook=False))
                analyse_model_size(model)
                train_with_synth_data(model, run_cfg={
                    **train_config["synth"],
                    "name": f"{model.__class__.__name__}_{n_kernels}_k_{block_size}_bs_synth"
                    }, num_blocks="6GB")
                finetune_with_real_data(model, run_cfg={
                    **train_config["real"],
                    "load_model_from": f"{model.__class__.__name__}_{n_kernels}_k_{block_size}_bs_synth_<latest>",
                    "name": f"{model.__class__.__name__}_{n_kernels}_k_{block_size}_bs_real_ft"
                    }, batch_size="6GB")
                del model
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    os.remove(tmp_file_path)
                except FileNotFoundError as e:
                    print("WARNING: FILE SHOULD EXIST")
                    print(e)