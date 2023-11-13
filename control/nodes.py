import numpy as np

import folder_paths

from .control import ControlNetAdvanced, T2IAdapterAdvanced, load_controlnet, ControlNetWeightsType, T2IAdapterWeightsType,\
    LatentKeyframeGroup, TimestepKeyframe, TimestepKeyframeGroup, is_advanced_controlnet
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


class AdvancedControlNetApply:
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
                "mask_optional": ("MASK", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/conditioning"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, mask_optional=None):
        if strength == 0:
            return (positive, negative)

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
                    # set cond hint mask
                    if mask_optional is not None:
                        if is_advanced_controlnet(c_net):
                            # if not in the form of a batch, make it so
                            if len(mask_optional.shape) < 3:
                                mask_optional = mask_optional.unsqueeze(0)
                            c_net.set_cond_hint_mask(mask_optional)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

class LinearBatchCreativeInterpolationNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                "images": ("IMAGE", ),
                "frames_per_keyframe": ("INT", {"default": 16, "min": 4, "max": 64, "step": 1}),
                "length_of_key_frame_influence": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.001}),
                "cn_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),      
                "soft_scaled_cn_weights_multiplier": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 10.0, "step": 0.01}),          
                "interpolation": (["ease-in", "ease-out", "ease-in-out"],),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "combined_function"

    CATEGORY = "ComfyUI-Creative-Interpolation 🎞️🅟🅞🅜/Batch"

    def combined_function(self, positive, negative, control_net_name, images, length_of_key_frame_influence,cn_strength,frames_per_keyframe,soft_scaled_cn_weights_multiplier,interpolation):
        
        def calculate_keyframe_peaks_and_influence(frames_per_keyframe, number_of_keyframes, length_of_influence, new_influence_range=(0,4)):
            number_of_frames = frames_per_keyframe * number_of_keyframes
            # Calculate the interval between keyframes
            interval = (number_of_frames - 1) // (number_of_keyframes - 1)
            # Determine if we need to adjust the interval because of a remainder
            adjustment = (number_of_frames - 1) % (number_of_keyframes - 1)

            # Calculate the peak frames for each keyframe
            peaks = [0]  # The first keyframe is always at the first frame
            for i in range(1, number_of_keyframes - 1):  # We already know the first and last keyframe peaks
                peak = peaks[-1] + interval
                # If we have a remainder, we distribute it among the first keyframes
                if i <= adjustment:
                    peak += 1
                peaks.append(peak)
            peaks.append(number_of_frames - 1)  # The last keyframe is always at the last frame

            # Calculate the full interval between keyframes
            full_interval = (number_of_frames - 1) / (number_of_keyframes - 1)
            # Calculate the scaled interval based on the length_of_influence
            scaled_interval = full_interval * length_of_influence

            # Initialize the list to store the influence range for each keyframe
            influence_ranges = [new_influence_range]

            # Shift the subsequent influence ranges
            shift = new_influence_range[1]

            # Loop through each keyframe to calculate its shifted influence range
            for i, peak in enumerate(peaks):
                # Calculate the start and end influence around the peak, shifted by the new range's end
                start_influence = max(shift, int(peak - scaled_interval / 2.0) + shift)
                end_influence = min(number_of_frames + shift, int(peak + scaled_interval / 2.0) + shift)
                # Add the influence range as a tuple (start, end) to the list
                influence_ranges.append((start_influence, end_influence))

            return influence_ranges
                        
        influence_ranges = calculate_keyframe_peaks_and_influence(frames_per_keyframe, len(images), length_of_key_frame_influence)

        for i, (start, end) in enumerate(influence_ranges):
            
            batch_index_from, batch_index_to_excl = influence_ranges[i]
                                                                                                
            if i == 0:  # buffer image
                image = images[0]
                strength_from = 1.0
                strength_to = 1.0            
                return_at_midpoint = False 
            elif i == 1: # First image
                image = images[0]
                strength_from = 1.0
                strength_to = 0.0
                return_at_midpoint = False                            
            elif i == len(images) - 1:  # Last image
                image = images[i-1]
                strength_from = 0.0
                strength_to = 1.0
                return_at_midpoint = False                                
            else:  # Middle images
                image = images[i-1]
                strength_from = 0.0
                strength_to = 1.0
                return_at_midpoint = True
                                                                                                              
            latent_keyframe_interpolation_node = LatentKeyframeInterpolationNode()
            latent_keyframe, = latent_keyframe_interpolation_node.load_keyframe(batch_index_from, strength_from, batch_index_to_excl, strength_to, interpolation,return_at_midpoint)                        
                
            scaled_soft_control_net_weights = ScaledSoftControlNetWeights()
            control_net_weights, _ = scaled_soft_control_net_weights.load_weights(soft_scaled_cn_weights_multiplier, False)

            timestep_keyframe_node = TimestepKeyframeNode()
            timestep_keyframe, = timestep_keyframe_node.load_keyframe(
                start_percent=0.0,
                control_net_weights=control_net_weights,
                t2i_adapter_weights=None,
                latent_keyframe=latent_keyframe,
                prev_timestep_keyframe=None
            )

            control_net_loader = ControlNetLoaderAdvanced()
            control_net, = control_net_loader.load_controlnet(control_net_name, timestep_keyframe)
                        
            apply_advanced_control_net = AdvancedControlNetApply()            
                        
            positive, negative = apply_advanced_control_net.apply_controlnet(positive, negative, control_net, image.unsqueeze(0), cn_strength, 0.0, 1.0)

        return (positive, negative)

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Combined
    "LinearBatchCreativeInterpolation": LinearBatchCreativeInterpolationNode
    # Keyframes
    # "TimestepKeyframe": TimestepKeyframeNode,
    # "LatentKeyframe": LatentKeyframeNode,
    # "LatentKeyframeGroup": LatentKeyframeGroupNode,
    # "LatentKeyframeBatchedGroup": LatentKeyframeBatchedGroupNode,
    # "LatentKeyframeTiming": LatentKeyframeInterpolationNode,
    # Loaders
    # "ControlNetLoaderAdvanced": ControlNetLoaderAdvanced,
    # "DiffControlNetLoaderAdvanced": DiffControlNetLoaderAdvanced,
    # Conditioning
    # "ACN_AdvancedControlNetApply": AdvancedControlNetApply,
    # Weights
    # "ScaledSoftControlNetWeights": ScaledSoftControlNetWeights,
    # "SoftControlNetWeights": SoftControlNetWeights,
    # "CustomControlNetWeights": CustomControlNetWeights,
    # "SoftT2IAdapterWeights": SoftT2IAdapterWeights,
    # "CustomT2IAdapterWeights": CustomT2IAdapterWeights,
    # Image
    # "LoadImagesFromDirectory": LoadImagesFromDirectory
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Combined
    "BatchCreativeInterpolation": "Batch Creative Interpolation 🎞️🅟🅞🅜"
    # Keyframes
    # "TimestepKeyframe": "Timestep Keyframe 🎞️🅟🅞🅜",
    # "LatentKeyframe": "Latent Keyframe 🛂🅐🅒🅝",
    # "LatentKeyframeGroup": "Latent Keyframe Group 🛂🅐🅒🅝",
    # "LatentKeyframeBatchedGroup": "Latent Keyframe Batched Group 🛂🅐🅒🅝",
    # "LatentKeyframeTiming": "Latent Keyframe Interpolation 🛂🅐🅒🅝",
    # Loaders
    # "ControlNetLoaderAdvanced": "Load ControlNet Model (Advanced) 🛂🅐🅒🅝",
    # "DiffControlNetLoaderAdvanced": "Load ControlNet Model (diff Advanced) 🛂🅐🅒🅝",
    # Conditioning
    # "ACN_AdvancedControlNetApply": "Apply Advanced ControlNet 🛂🅐🅒🅝",
    # Weights
    # "ScaledSoftControlNetWeights": "Scaled Soft ControlNet Weights 🛂🅐🅒🅝",
    # "SoftControlNetWeights": "Soft ControlNet Weights 🛂🅐🅒🅝",
    # "CustomControlNetWeights": "Custom ControlNet Weights 🛂🅐🅒🅝",
    # "SoftT2IAdapterWeights": "Soft T2IAdapter Weights 🛂🅐🅒🅝",
    # "CustomT2IAdapterWeights": "Custom T2IAdapter Weights 🛂🅐🅒🅝",
    # Image
    # "LoadImagesFromDirectory": "Load Images [DEPRECATED] 🛂🅐🅒🅝"
}
