from abc import abstractmethod
import json
import os
from typing import Any, Dict, Tuple, TypeVar
import torch
from deepdiff.diff import DeepDiff
T = TypeVar('T', bound='SmoeModel')

__all__ = [
    "SmoeModel"
]

class SmoeModel(torch.nn.Module):
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

    def save_state_dict(self: T, path: str) -> None:
        """Saves the model's state dict in path. If the path is not findable, store it in the same folder
        where the model's source code and configs are

        Args:
            path (str): Either a full path or a file name
        """
        if os.path.isdir(os.path.dirname(path)):
            torch.save(self.state_dict(), path)
            return
        current_path = os.path.join(self.saves_path, path)
        os.makedirs(current_path, exist_ok=True)
        with open(os.path.join(current_path, "model_configs.json"), "w") as f:
            json.dump(self.cfg, f)
        torch.save(self.state_dict(), os.path.join(current_path, "state_dict.pth"))

    def load_state_dict(self: T, path: str) -> None:
        """Loads the model from path. Path can be just a file name if it's stored in the saves folder

        Args:
            path (str): Full path or name of file in saves directory
        """
        if os.path.isdir(os.path.dirname(path)):
            self.load_state_dict(torch.load(path))
            return
        current_path = os.path.join(self.saves_path, path)
        with open(os.path.join(current_path, "model_configs.json"), "r") as f:
            cfg = json.load(f)
        diff = DeepDiff(self.cfg, cfg)
        if len(diff) > 0:
            print("Model configuration and loaded configuration do not match.\n")
            print(diff)
        self.load_state_dict(torch.load(os.path.join(current_path, "state_dict.pth")))
