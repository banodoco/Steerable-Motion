from torch import Tensor

import folder_paths
from nodes import VAEEncode
import comfy.utils

# from .utils import TimestepKeyframeGroup
from .control_sparsectrl import SparseIndexMethodImport
# from .control import load_sparsectrl, load_controlnet, ControlNetAdvanced, SparseCtrlAdvanced



class SparseIndexMethodNodeImport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "indexes": ("STRING", {"default": "0"}),
            }
        }
    
    RETURN_TYPES = ("SPARSE_METHOD",)
    FUNCTION = "get_method"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl"

    def get_method(self, indexes: str):
        idxs = []
        unique_idxs = set()
        # get indeces from string
        str_idxs = [x.strip() for x in indexes.strip().split(",")]
        for str_idx in str_idxs:
            try:
                idx = int(str_idx)
                if idx in unique_idxs:
                    raise ValueError(f"'{idx}' is duplicated; indexes must be unique.")
                idxs.append(idx)
                unique_idxs.add(idx)
            except ValueError:
                raise ValueError(f"'{str_idx}' is not a valid integer index.")
        if len(idxs) == 0:
            raise ValueError(f"No indexes were listed in Sparse Index Method.")
        return (SparseIndexMethodImport(idxs),)

