import numpy as np

import folder_paths

from .control import ControlNetAdvanced, T2IAdapterAdvanced, load_controlnet, ControlNetWeightsType, T2IAdapterWeightsType,\
    LatentKeyframeGroup, TimestepKeyframe, TimestepKeyframeGroup
from .weight_nodes import ScaledSoftControlNetWeights, SoftControlNetWeights, CustomControlNetWeights, \
    SoftT2IAdapterWeights, CustomT2IAdapterWeights
from .latent_keyframe_nodes import LatentKeyframeGroupNode, LatentKeyframeInterpolationNode, LatentKeyframeBatchedGroupNode, LatentKeyframeNode
from .deprecated_nodes import LoadImagesFromDirectory
from .logger import logger


class TimestepKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "control_net_weights": ("CONTROL_NET_WEIGHTS", ),
                "t2i_adapter_weights": ("T2I_ADAPTER_WEIGHTS", ),
                "latent_keyframe": ("LATENT_KEYFRAME", ),
                "prev_timestep_keyframe": ("TIMESTEP_KEYFRAME", ),
            }
        }
    
    RETURN_TYPES = ("TIMESTEP_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                      start_percent: float,
                      control_net_weights: ControlNetWeightsType=None,
                      t2i_adapter_weights: T2IAdapterWeightsType=None,
                      latent_keyframe: LatentKeyframeGroup=None,
                      prev_timestep_keyframe: TimestepKeyframeGroup=None):
        if not prev_timestep_keyframe:
            prev_timestep_keyframe = TimestepKeyframeGroup()
        keyframe = TimestepKeyframe(start_percent, control_net_weights, t2i_adapter_weights, latent_keyframe)
        prev_timestep_keyframe.add(keyframe)
        return (prev_timestep_keyframe,)


class ControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            },
            "optional": {
                "timestep_keyframe": ("TIMESTEP_KEYFRAME", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/loaders"

    def load_controlnet(self, control_net_name, timestep_keyframe: TimestepKeyframeGroup=None):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe)
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
                "timestep_keyframe": ("TIMESTEP_KEYFRAME", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/loaders"

    def load_controlnet(self, control_net_name, timestep_keyframe: TimestepKeyframeGroup, model):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe, model)
        return (controlnet,)


class ControlNetApplyAdvanced_AdvControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "mask_opt": ("MASK", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/loaders/conditioning"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, mask_opt=None):
        if strength == 0:
            return (positive, negative)

        if mask_opt is not None:
            mask_hint = mask_opt.movedim(-1,1)
        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (1.0 - start_percent, 1.0 - end_percent))
                    # TODO: finish mask implemention, does nothing right now
                    if mask_opt is not None:
                        if isinstance(c_net, ControlNetAdvanced) or isinstance(c_net, T2IAdapterAdvanced):
                            c_net.set_cond_hint_mask(mask_hint)
                        else:
                            logger
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": TimestepKeyframeNode,
    "LatentKeyframe": LatentKeyframeNode,
    "LatentKeyframeGroup": LatentKeyframeGroupNode,
    "LatentKeyframeBatchedGroup": LatentKeyframeBatchedGroupNode,
    "LatentKeyframeTiming": LatentKeyframeInterpolationNode,
    # Loaders
    "ControlNetLoaderAdvanced": ControlNetLoaderAdvanced,
    "DiffControlNetLoaderAdvanced": DiffControlNetLoaderAdvanced,
    # Weights
    "ScaledSoftControlNetWeights": ScaledSoftControlNetWeights,
    "SoftControlNetWeights": SoftControlNetWeights,
    "CustomControlNetWeights": CustomControlNetWeights,
    "SoftT2IAdapterWeights": SoftT2IAdapterWeights,
    "CustomT2IAdapterWeights": CustomT2IAdapterWeights,
    # Image
    "LoadImagesFromDirectory": LoadImagesFromDirectory
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": "Timestep Keyframe 🛂🅐🅒🅝",
    "LatentKeyframe": "Latent Keyframe 🛂🅐🅒🅝",
    "LatentKeyframeGroup": "Latent Keyframe Group 🛂🅐🅒🅝",
    "LatentKeyframeBatchedGroup": "Latent Keyframe Batched Group 🛂🅐🅒🅝",
    "LatentKeyframeTiming": "Latent Keyframe Interpolation 🛂🅐🅒🅝",
    # Loaders
    "ControlNetLoaderAdvanced": "Load ControlNet Model (Advanced) 🛂🅐🅒🅝",
    "DiffControlNetLoaderAdvanced": "Load ControlNet Model (diff Advanced) 🛂🅐🅒🅝",
    # Weights
    "ScaledSoftControlNetWeights": "Scaled Soft ControlNet Weights 🛂🅐🅒🅝",
    "SoftControlNetWeights": "Soft ControlNet Weights 🛂🅐🅒🅝",
    "CustomControlNetWeights": "Custom ControlNet Weights 🛂🅐🅒🅝",
    "SoftT2IAdapterWeights": "Soft T2IAdapter Weights 🛂🅐🅒🅝",
    "CustomT2IAdapterWeights": "Custom T2IAdapter Weights 🛂🅐🅒🅝",
    # Image
    "LoadImagesFromDirectory": "Load Images [DEPRECATED] 🛂🅐🅒🅝"
}
