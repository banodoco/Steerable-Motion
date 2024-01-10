#taken from: https://github.com/lllyasviel/ControlNet
#and modified
#and then taken from comfy/cldm/cldm.py and modified again

from abc import ABC, abstractmethod
import math
import numpy as np
from typing import Iterable, Union
import torch
import torch as th
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat

from comfy.ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)

from comfy.cldm.cldm import ControlNet as ControlNetCLDM
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample
from comfy.ldm.util import exists
from comfy.ldm.modules.attention import default, optimized_attention
from comfy.ldm.modules.attention import FeedForward, SpatialTransformer
from comfy.controlnet import broadcast_image_to
from comfy.utils import repeat_to_batch_size
import comfy.ops

# from .utils import TimestepKeyframeGroup, disable_weight_init_clean_groupnorm, prepare_mask_batch





class SparseMethodImport(ABC):
    SPREAD = "spread"
    INDEX = "index"
    def __init__(self, method: str):
        self.method = method

    @abstractmethod
    def get_indexes(self, hint_length: int, full_length: int) -> list[int]:
        pass



class SparseIndexMethodImport(SparseMethodImport):
    def __init__(self, idxs: list[int]):
        super().__init__(self.INDEX)
        self.idxs = idxs

    def get_indexes(self, hint_length: int, full_length: int) -> list[int]:
        orig_hint_length = hint_length
        if hint_length > full_length:
            hint_length = full_length
        # if idxs is less than hint_length, throw error
        if len(self.idxs) < hint_length:
            err_msg = f"There are not enough indexes ({len(self.idxs)}) provided to fit the usable {hint_length} input images."
            if orig_hint_length != hint_length:
                err_msg = f"{err_msg} (original input images: {orig_hint_length})"
            raise ValueError(err_msg)
        # cap idxs to hint_length
        idxs = self.idxs[:hint_length]
        new_idxs = []
        real_idxs = set()
        for idx in idxs:
            if idx < 0:
                real_idx = full_length+idx
                if real_idx in real_idxs:
                    raise ValueError(f"Index '{idx}' maps to '{real_idx}' and is duplicate - indexes in Sparse Index Method must be unique.")
            else:
                real_idx = idx
                if real_idx in real_idxs:
                    raise ValueError(f"Index '{idx}' is duplicate (or a negative index is equivalent) - indexes in Sparse Index Method must be unique.")
            real_idxs.add(real_idx)
            new_idxs.append(real_idx)
        return new_idxs

