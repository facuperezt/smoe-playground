import pickle
from typing import Literal, Optional
import torch 
from torchvision.transforms import v2, ToTensor, Grayscale
import os
from PIL import Image
import tqdm
from itertools import cycle

from .kernel_space_generator import get_m_samples
import threading
import queue

from src.models.components.decoders import SmoeDecoder

def initialize_transforms(img_size: int = 512):
    transforms = v2.Compose([
        ToTensor(),
        Grayscale(),
        v2.RandomCrop(size=(img_size, img_size)),
        v2.RandomResizedCrop(size=(img_size, img_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=False),
    ])

    return transforms

class DataLoader:
    _dataloader_path = os.path.realpath(__file__).split("dataloader.py")[0]

    def __init__(self, mode: Literal["dataset", "synthetic"], n_kernels: int = 4, block_size: int = 16, data_path: str = "", img_size: int = 512, n_repeats: int = 5,
                 force_reinitialize: bool = False, batch_size: int = 25, kernels_outside: bool = False,
                 negative_experts: bool = False, device: torch.device = "cuda"):
        self.batch_size = batch_size
        self.n_kernels = n_kernels
        self.kernels_outside = kernels_outside
        self.negative_experts = negative_experts
        self.device = device
        self.block_size = block_size
        self.img_size = img_size
        self.transforms = initialize_transforms(img_size)
        self.training_data = []
        self.validation_data = []
        self.training_data_path = os.path.join(os.path.realpath(__file__).split("dataloader.py")[0], data_path, "train")
        self.validation_data_path = os.path.join(os.path.realpath(__file__).split("dataloader.py")[0], data_path, "valid")
        if mode == "dataset":
            if data_path == "":
                raise ValueError("Need to provide a path to the dataset.")
            self.initialized = False
            self.initialize(n_repeats, force_reinitialize)
            self._dataset_train = cycle(self._get("train", None))
            self.mode = "dataset"
        else:
            self.mode = "synthetic"
            self.decoder = SmoeDecoder(n_kernels, block_size, device)

    @property
    def training(self):
        return self.get("train", None, 3)

    def get_valid_pic(self):
        return self.transforms(Image.open(os.path.join(self.validation_data_path, os.pardir, "sample_comparisson_photo.png")).convert("RGB"))

    @staticmethod
    def generate_samples(q: queue.Queue, n_batches, n_blocks, n_kernels, kernels_outside: bool = False, negative_experts: bool = False,
                         vmin: float = -5., vmax: float = 5, include_zero=False):
        for _ in range(n_batches):
            samples = get_m_samples(n_blocks, n_kernels, kernels_outside=kernels_outside, negative_experts=negative_experts, vmin=vmin, vmax=vmax, include_zero=include_zero, device="cpu")
            q.put(samples)

    def generate_random_blocks_queue(self, n_batches: int, n_blocks: int, n_kernels: int, block_size: int, include_zero: bool = False, negative_experts: bool = False):
        q = queue.Queue()
        thread = threading.Thread(target=self.generate_samples, args=(q, n_batches, n_blocks, n_kernels, block_size, include_zero, negative_experts), daemon=True)
        thread.start()
        return q
    
    def get_m_blocks_with_n_kernels(self, m: int, n: Optional[int] = None, vmin: float = -5, vmax: float = 5, include_zero: bool = True):
        if n is None:
            n = self.n_kernels
        out = get_m_samples(m, n, kernels_outside=self.kernels_outside, negative_experts=self.negative_experts, vmin=vmin, vmax=vmax, include_zero=include_zero, device=self.device)
        return out

    def _get(self, data: str = "train", limit_to: int = None):
        if data == "train":
            data = torch.cat(self.training_data[:limit_to], dim=0)
        elif data == "valid":
            data = torch.cat(self.validation_data[:limit_to], dim=0)
        else:
            raise ValueError("train or valid expected")
        perm = torch.randperm(len(data))
        data = data[perm][:, None, :, :]
        if self.batch_size == 0:
            self.batch_size = len(data)
        elif self.batch_size < 0:
            self.batch_size = max(1, int(len(data)/(-self.batch_size)))
        for i in range(0, len(data), self.batch_size):
            yield data[i:i+self.batch_size]
        yield data[i+self.batch_size:]

    def get(self, *args, **kwargs):
        if self.mode == "synthetic":
            out =  self.get_m_blocks_with_n_kernels(*args, **kwargs)
            return self.decoder(out).transpose(1,0)
        elif self.mode == "dataset":
            out: torch.Tensor = next(self._dataset_train)
            if out.numel() == 0:
                out = next(self._dataset_train)
            if out.numel() == 0:
                raise ValueError("Iterator be trippin")
            return out

    def initialize(self, n_repeats: int = 3, force_reinitialize: bool = False) -> None:
        train_pkl_not_found = not os.path.exists(f"{self.training_data_path}/train.pkl")
        valid_pkl_not_found = not os.path.exists(f"{self.validation_data_path}/valid.pkl")
        if force_reinitialize:
            self.fill_training_data(use_saved=False, n_repeats=n_repeats)
            self.fill_validation_data(use_saved=False, n_repeats=n_repeats)
            self.initialized = True
            return
        
        if train_pkl_not_found:
            self.fill_training_data(use_saved=False, n_repeats=n_repeats)
        else:
            self.fill_training_data(n_repeats=n_repeats)
        if valid_pkl_not_found:
            self.fill_validation_data(use_saved=False, n_repeats=n_repeats)
        else:
            self.fill_validation_data(n_repeats=n_repeats)
        if self.training_data[0].shape[-1] != self.img_size:
            self.training_data = []
            self.validation_data = []
            self.initialize(n_repeats, force_reinitialize=True)
        self.initialized = True
        

    def fill_training_data(self, use_saved: bool = True, n_repeats: int = 3) -> None:
        if use_saved:
            with open(f"{self.training_data_path}/train.pkl", "rb") as f:
                self.training_data = pickle.load(f)
                return
            
        for image_path in tqdm.tqdm(os.listdir(self.training_data_path), "Filling Training Set: "):
            if image_path.startswith(".") or image_path.endswith(".pkl"):
                continue
            img = Image.open(os.path.join(self.training_data_path, image_path))
            try:
                self.training_data.extend([self.transforms(_img) for _img in n_repeats*[img]])
            except ValueError as e:
                print(e)
                continue
        if not use_saved:
            with open(f"{self.training_data_path}/train.pkl", "wb") as f:
                pickle.dump(self.training_data, f)

    def fill_validation_data(self, use_saved: bool = True, n_repeats: int = 3) -> None:
        if use_saved:
            with open(f"{self.validation_data_path}/valid.pkl", "rb") as f:
                self.validation_data = pickle.load(f)
                return
        for image_path in tqdm.tqdm(os.listdir(self.validation_data_path), "Filling Test Set: "):
            if image_path.startswith(".") or image_path.endswith(".pkl"):
                continue
            img = Image.open(os.path.join(self.validation_data_path, image_path))
            try:
                self.validation_data.extend([self.transforms(_img) for _img in n_repeats*[img]])
            except ValueError as e:
                print(e)
                continue
        if not use_saved:
            with open(f"{self.validation_data_path}/valid.pkl", "wb") as f:
                pickle.dump(self.validation_data, f)

    def get_epoch_training_data(self):
        for x, y in zip(self.training_data, self.training_data):
            yield x, y

    def get_epoch_validation_data(self):
        for x, y in zip(self.validation_data, self.validation_data):
            yield x, y

    # def get_epoch_validation_data(self):
    #     for x, y in zip(self.validation_data, self.validation_data):
    #         yield torch.tensor(sliding_window(x.squeeze().numpy(), 2*[self.block_size], 2*[self.block_size], flatten=False), dtype=torch.float32, requires_grad=True), torch.tensor(sliding_window(y.squeeze().numpy(), 2*[self.block_size], 2*[self.block_size], flatten=False), dtype=torch.float32, requires_grad=True)
        