import sys
import os

import folder_paths

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
from comfy.sd import ControlBase

from .control import load_controlnet, ControlNetWeightsType, T2IAdapterWeightsType,\
    LatentKeyframe, LatentKeyframeGroup, TimestepKeyframe, TimestepKeyframeGroup


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

    CATEGORY = "adv-controlnet/weights"

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

    CATEGORY = "adv-controlnet/weights"

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

    CATEGORY = "adv-controlnet/weights"

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

    CATEGORY = "adv-controlnet/weights"

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
                "weight_00": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ([False, True], ),
            },
        }
    
    RETURN_TYPES = ("T2I_ADAPTER_WEIGHTS", )
    FUNCTION = "load_weights"

    CATEGORY = "adv-controlnet/weights"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03]
        if flip_weights:
            weights.reverse()
        return (weights,)


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

    CATEGORY = "adv-controlnet/keyframes"

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
    

class LatentKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.00001}, ),
            },
            "optional": {
                "prev_latent_keyframe": ("LATENT_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "adv-controlnet/keyframes"

    def load_keyframe(self,
                      batch_index: int,
                      strength: float,
                      prev_latent_keyframe: LatentKeyframeGroup=None):
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        keyframe = LatentKeyframe(batch_index, strength)
        prev_latent_keyframe.add(keyframe)
        return (prev_latent_keyframe,)


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

    CATEGORY = "adv-controlnet/loaders"

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
                "control_net_weights": ("CONTROL_NET_WEIGHTS", ),
                "t2i_adapter_weights": ("T2I_ADAPTER_WEIGHTS", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "adv-controlnet/loaders"

    def load_controlnet(self, control_net_name, timestep_keyframe: TimestepKeyframeGroup, model):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe, model)
        return (controlnet,)


class ControlNetApplyPartialBatch: # NOT USED: was used for a different test, has useful index parsing code though
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
                "latent_image": ("LATENT", ),
                "latent_indeces": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "adv-controlnet/conditioning"

    def validate_index(self, index: int, latent_count: int, is_range: bool = False) -> int:
        # if part of range, do nothing
        if is_range:
            return index
        # otherwise, validate index
        # validate not out of range
        if index > latent_count-1:
            raise IndexError(f"Index '{index}' out of range for the total {latent_count} latents.")
        # if negative, validate not out of range
        if index < 0:
            conv_index = latent_count+index
            if conv_index < 0:
                raise IndexError(f"Index '{index}', converted to '{conv_index}' out of range for the total {latent_count} latents.")
            index = conv_index
        return index

    def convert_to_index_int(self, raw_index: str, is_range: bool = False) -> int:
        try:
            return self.validate_index(int(raw_index), is_range=is_range)
        except ValueError as e:
            raise ValueError(f"index '{raw_index}' must be an integer.", e)

    def convert_to_indeces(self, latent_indeces: str, latent_count: int) -> set[int]:
        if not latent_indeces:
            return set()
        all_indeces = [i for i in range(0, latent_count)]
        chosen_indeces = set()
        # parse string - allow positive ints, negative ints, and ranges separated by ':'
        groups = latent_indeces.split(",")
        groups = [g.strip() for g in groups]
        for g in groups:
            # parse range of indeces (e.g. 2:16)
            if ':' in g:
                index_range = g.split(":", 1)
                index_range = [r.strip() for r in index_range]
                start_index = self.convert_to_index_int(index_range[0], is_range=True)
                end_index = self.convert_to_index_int(index_range[1], is_range=True)
                for i in all_indeces[start_index, end_index]:
                    chosen_indeces.add(i)
            # parse individual indeces
            else:
                chosen_indeces.add(self.convert_to_index_int(g))
        return chosen_indeces

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, latent_image=None, latent_indeces: str=None):
        if strength == 0:
            return (positive, negative)

        latent_count = 1
        if latent_image:
            latent_count = latent_image['samples'].size()[0]
        indeces_to_apply = self.convert_to_indeces(latent_indeces, latent_count)

        control_hint = image.movedim(-1,1)
        cnets = {}

        evaluating_positive = True
        out = []
        for conditioning in [positive, negative]:
            c = []
            if evaluating_positive and latent_count > 1:
                # should copy positive conditioning to match latent_count
                if len(conditioning) < latent_count:
                    pass
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (1.0 - start_percent, 1.0 - end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
                evaluating_positive = False
            out.append(c)
        return (out[0], out[1])


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Keyframes
    "TimestepKeyframe": TimestepKeyframeNode,
    "LatentKeyframe": LatentKeyframeNode,
    # Conditioning
    # "ControlNetApplyPartialBatch": ControlNetApplyPartialBatch,
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
    # Keyframes
    "TimestepKeyframe": "Timestep Keyframe",
    "LatentKeyframe": "Latent Keyframe",
    # Conditioning
    # "ControlNetApplyPartialBatch": "Apply ControlNet (Partial Batch)",
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
