from abc import abstractmethod
import json
import os
from typing import Any, Dict, Optional, Tuple, TypeVar
import torch
from deepdiff.diff import DeepDiff
T = TypeVar('T', bound='SmoeModel')

__all__ = [
    "SmoeModel"
]

class SmoeModel(torch.nn.Module):
    def __init__(self: T):
        super().__init__()
        self._saved_models_no_args = 0

    @property
    def cfg(self: T) -> Dict[str, Any]:
        """Dictionary containing all of the models used configs.
        Normally this is a nested dict with a dict for the parameters
        of each part of the model

        Returns:
            Dict[str, Any]: Model's parameters
        """
        return self._cfg
    
    @property
    def saves_path(self: T) -> str:
        return self._saves_path

    @property
    def num_params(self: T) -> int:
        return sum(p.numel() for p in self.parameters())

    @abstractmethod
    def loss(self: T, input: torch.Tensor, output: Any, extra_information: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Each model implements it's own loss function using the full output of it's forward
        method as the second argument and any extra information (e.g. like from the data generation)
        passed as the third argument

        Args:
            input (torch.Tensor): The input to the forward function
            output (Any): the output of the forward function
            extra_information (Any): Any extra information that can be used by the loss function

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: A tuple containing the loss tensor as the first element and any information that needs to be logged as the second

        """ 
        pass

    @abstractmethod
    def reconstruct_input(self: T, input: torch.Tensor) -> torch.Tensor:
        """Implements a forward call that only returns the reconstructed input. Used to validate models and check reconstruction quality in
        a general way.

        Args:
            input (torch.Tensor): Input to the model

        Returns:
            torch.Tensor: The encoded and decoded (reconstructed) input.
        """
        pass

    def save_model(self: T, path: Optional[str] = None) -> str:
        """Saves the model's state dict in path. If the path is not findable, store it in the same folder
        where the model's source code and configs are

        Args:
            path (str): Either a full path or a file name

        Return:
            str: The path that was found to store the model.
        """
        if path is None:
            path = os.path.join("current_run", str(self._saved_models_no_args))
            self._saved_models_no_args += 1
        elif os.path.isdir(os.path.dirname(path)):
            torch.save(self.state_dict(), path)
            return path
        current_path = os.path.join(self.saves_path, path)
        os.makedirs(current_path, exist_ok=True)
        with open(os.path.join(current_path, "model_configs.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)
        torch.save(self.state_dict(), os.path.join(current_path, "state_dict.pth"))
        return current_path
    
    def load_model(self: T, path: str) -> None:
        """Loads the model from path. Path can be just a file name if it's stored in the saves folder

        Args:
            path (str): Full path or name of file in saves directory
        """
        if os.path.isdir(os.path.dirname(path)):
            self.load_state_dict(torch.load(os.path.join(path, "state_dict.pth")))
            return
        current_path = os.path.join(self.saves_path, path)
        if "<latest>" in path:
            dirname = os.path.dirname(current_path)
            all_models_in_dir = os.listdir(dirname)
            filtered_models_in_dir = [save for save in all_models_in_dir if save.startswith(path.split("<latest>")[0])]
            latest_matching_model = filtered_models_in_dir[-1]
            current_path = os.path.join(dirname, latest_matching_model)

        with open(os.path.join(current_path, "model_configs.json"), "r") as f:
            cfg = json.load(f)
        diff = DeepDiff(self.cfg, cfg)
        if len(diff) > 0:
            print("Model configuration and loaded configuration do not match.\n")
            print(diff)
        try:
            self.load_state_dict(torch.load(os.path.join(current_path, "state_dict.pth")))
        except RuntimeError as e:
            print(f"PyTorch failed to load the model, original error message:\n\n{e}\n\n")
            raise IOError(f"File {os.path.join(current_path, 'state_dict.pth')} is most likely corrupted and cannot be read correctly by PyTorch.")
