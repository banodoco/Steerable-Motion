import sys
import os

import folder_paths

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
from comfy.sd import ControlBase

from .control import load_controlnet, ControlNetWeightsType, T2IAdapterWeightsType


class ScaledSoftControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_multiplier": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", )
    FUNCTION = "load_weights"

    CATEGORY = "controlnet weights"

    def load_weights(self, base_multiplier, flip_weights):
        weights = [(base_multiplier ** float(12 - i)) for i in range(13)]
        if flip_weights:
            weights.reverse()
        return (weights, )


class SoftControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.09941396206337118, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.12050177219802567, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.14606275417942507, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 0.17704576264172736, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_04": ("FLOAT", {"default": 0.214600924414215, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_05": ("FLOAT", {"default": 0.26012233262329093, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_06": ("FLOAT", {"default": 0.3152997971191405, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_07": ("FLOAT", {"default": 0.3821815722656249, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_08": ("FLOAT", {"default": 0.4632503906249999, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_09": ("FLOAT", {"default": 0.561515625, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_10": ("FLOAT", {"default": 0.6806249999999999, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_11": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", )
    FUNCTION = "load_weights"

    CATEGORY = "controlnet weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11, weight_12]
        if flip_weights:
            weights.reverse()
        return (weights,)


class CustomControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_04": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_05": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_06": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_07": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_08": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_09": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_10": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_11": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", )
    FUNCTION = "load_weights"

    CATEGORY = "controlnet weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11, weight_12]
        if flip_weights:
            weights.reverse()
        return (weights,)


class SoftT2IAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.62, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("T2I_ADAPTER_WEIGHTS", )
    FUNCTION = "load_weights"

    CATEGORY = "controlnet weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03]
        if flip_weights:
            weights.reverse()
        return (weights,)


class CustomT2IAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.62, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("T2I_ADAPTER_WEIGHTS", )
    FUNCTION = "load_weights"

    CATEGORY = "controlnet weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03]
        if flip_weights:
            weights.reverse()
        return (weights,)


class ControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            },
            "optional": {
                "control_net_weights": ("CONTROL_NET_WEIGHTS", ),
                "t2i_adapter_weights": ("T2I_ADAPTER_WEIGHTS", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name, control_net_weights: ControlNetWeightsType=None, t2i_adapter_weights: T2IAdapterWeightsType=None):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, control_net_weights, t2i_adapter_weights)
        return (controlnet,)
    

class DiffControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), )
            },
            "optional": {
                "control_net_weights": ("CONTROL_NET_WEIGHTS", ),
                "t2i_adapter_weights": ("T2I_ADAPTER_WEIGHTS", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name, control_net_weights, t2i_adapter_weights, model):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, control_net_weights, t2i_adapter_weights, model)
        return (controlnet,)


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Loaders
    "ControlNetLoaderAdvanced": ControlNetLoaderAdvanced,
    "DiffControlNetLoaderAdvanced": DiffControlNetLoaderAdvanced,
    # Weights
    "ScaledSoftControlNetWeights": ScaledSoftControlNetWeights,
    "SoftControlNetWeights": SoftControlNetWeights,
    "CustomControlNetWeights": CustomControlNetWeights,
    "SoftT2IAdapterWeights": SoftT2IAdapterWeights,
    "CustomT2IAdapterWeights": CustomT2IAdapterWeights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "ControlNetLoaderAdvanced": "Load ControlNet Model (Advanced)",
    "DiffControlNetLoaderAdvanced": "Load ControlNet Model (diff Advanced)",
    # Weights
    "ScaledSoftControlNetWeights": "Scaled Soft ControlNet Weights",
    "SoftControlNetWeights": "Soft ControlNet Weights",
    "CustomControlNetWeights": "Custom ControlNet Weights",
    "SoftT2IAdapterWeights": "Soft T2IAdapter Weights",
    "CustomT2IAdapterWeights": "Custom T2IAdapter Weights",
}
