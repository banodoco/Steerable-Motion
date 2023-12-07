import numpy as np
import torch
import folder_paths
from PIL import Image
from ast import literal_eval
from .control import ControlNetAdvancedImport, T2IAdapterAdvancedImport, load_controlnet, ControlNetWeightsTypeImport, T2IAdapterWeightsTypeImport,\
    LatentKeyframeGroupImport, TimestepKeyframeImport, TimestepKeyframeGroupImport, is_advanced_controlnet
import matplotlib.pyplot as plt
from .IPAdapterPlus import contrast_adaptive_sharpening, IPAdapterApply,prep_image

from .weight_nodes import ScaledSoftControlNetWeightsImport, SoftControlNetWeightsImport, CustomControlNetWeightsImport, \
    SoftT2IAdapterWeightsImport, CustomT2IAdapterWeightsImport
from .latent_keyframe_nodes import LatentKeyframeGroupNodeImport, LatentKeyframeInterpolationNodeImport, LatentKeyframeBatchedGroupNodeImport, LatentKeyframeNodeImport,calculate_weights
from .deprecated_nodes import LoadImagesFromDirectory
from .logger import logger
import torchvision.transforms as TT
import torch.nn.functional as F

import comfy.utils
import comfy.model_management
from comfy.clip_vision import clip_preprocess
from comfy.ldm.modules.attention import optimized_attention
# import BytesIO
from io import BytesIO



class TimestepKeyframeNodeImport:
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
                      control_net_weights: ControlNetWeightsTypeImport=None,
                      t2i_adapter_weights: T2IAdapterWeightsTypeImport=None,
                      latent_keyframe: LatentKeyframeGroupImport=None,
                      prev_timestep_keyframe: TimestepKeyframeGroupImport=None):
        if not prev_timestep_keyframe:
            prev_timestep_keyframe = TimestepKeyframeGroupImport()
        keyframe = TimestepKeyframeImport(start_percent, control_net_weights, t2i_adapter_weights, latent_keyframe)
        prev_timestep_keyframe.add(keyframe)
        return (prev_timestep_keyframe,)


class ControlNetLoaderAdvancedImport:
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

    def load_controlnet(self, control_net_name, timestep_keyframe: TimestepKeyframeGroupImport=None):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe)
        return (controlnet,)
    

class DiffControlNetLoaderAdvancedImport:
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

    def load_controlnet(self, control_net_name, timestep_keyframe: TimestepKeyframeGroupImport, model):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, timestep_keyframe, model)
        return (controlnet,)


class AdvancedControlNetApplyImport:
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
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
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
    

class MaskGeneratorNode:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_masks"
    CATEGORY = "Steerable-Motion/Interpolation"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "number_of_masks": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
            },
        }

    def generate_masks(self, number_of_masks, strength, width, height):

        masks = []
        for _ in range(number_of_masks):
            mask = torch.full((height, width), strength)
            masks.append(mask)

        # Convert list of masks to a single tensor
        masks_tensor = torch.stack(masks, dim=0)
        return masks_tensor



class BatchCreativeInterpolationNode:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "images": ("IMAGE", ),
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),                
                "type_of_frame_distribution": (["linear", "dynamic"],),
                "linear_frame_distribution_value": ("INT", {"default": 16, "min": 4, "max": 64, "step": 1}),                
                "dynamic_frame_distribution_values": ("STRING", {"multiline": True, "default": "0,10,26,40"}),
                "type_of_key_frame_influence": (["linear", "dynamic"],),
                "linear_key_frame_influence_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "dynamic_key_frame_influence_values": ("STRING", {"multiline": True, "default": "1.0,1.0,1.0,0.5"}),
                "type_of_cn_strength_distribution": (["linear", "dynamic"],),
                "linear_cn_strength_value": ("STRING", {"multiline": False, "default": "(0.0,0.4)"}),
                "dynamic_cn_strength_values": ("STRING", {"multiline": True, "default": "(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)"}),
                "soft_scaled_cn_weights_multiplier": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 10.0, "step": 0.1}),          
                # "interpolation": (["ease-in-out", "ease-in", "ease-out"],),
                "buffer": ("INT", {"default": 4, "min": 0, "max": 16, "step": 1}),
                "relative_ipadapter_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "relative_ipadapter_influence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "ipadapter_noise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE","CONDITIONING","CONDITIONING","MODEL",)
    RETURN_NAMES = ("GRAPH","POSITIVE", "NEGATIVE","MODEL")
    FUNCTION = "combined_function"

    CATEGORY = "Steerable-Motion/Interpolation"

    def combined_function(self, positive, negative, images,model,ipadapter,clip_vision,control_net_name, 
                          type_of_frame_distribution,linear_frame_distribution_value,dynamic_frame_distribution_values,
                          type_of_key_frame_influence,linear_key_frame_influence_value,dynamic_key_frame_influence_values,
                          type_of_cn_strength_distribution,linear_cn_strength_value,dynamic_cn_strength_values,
                          soft_scaled_cn_weights_multiplier,buffer,relative_ipadapter_strength,
                          relative_ipadapter_influence,ipadapter_noise):
        
        def calculate_dynamic_influence_ranges(keyframe_positions, key_frame_influence_values, allow_extension=True):
            if len(keyframe_positions) < 2 or len(keyframe_positions) != len(key_frame_influence_values):
                return []

            influence_ranges = []
            for i, position in enumerate(keyframe_positions):
                influence_factor = key_frame_influence_values[i]

                # Calculate the base range size
                range_size = influence_factor * (keyframe_positions[-1] - keyframe_positions[0]) / (len(keyframe_positions) - 1) / 2

                # Calculate symmetric start and end influence
                start_influence = position - range_size
                end_influence = position + range_size

                # Adjust start and end influence to not exceed previous and next keyframes
                if not allow_extension:
                    start_influence = max(start_influence, keyframe_positions[i - 1] if i > 0 else 0)
                    end_influence = min(end_influence, keyframe_positions[i + 1] if i < len(keyframe_positions) - 1 else keyframe_positions[-1])

                influence_ranges.append((round(start_influence), round(end_influence)))

            return influence_ranges
            
        def add_starting_buffer(influence_ranges, buffer=4):
            shifted_ranges = [(0, buffer)]
            for start, end in influence_ranges:
                shifted_ranges.append((start + buffer, end + buffer))
            return shifted_ranges
        
        def get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, images, linear_frame_distribution_value):
            if type_of_frame_distribution == "dynamic":
                # Check if the input is a string or a list
                if isinstance(dynamic_frame_distribution_values, str):
                    # Sort the keyframe positions in numerical order
                    return sorted([int(kf.strip()) for kf in dynamic_frame_distribution_values.split(',')])
                elif isinstance(dynamic_frame_distribution_values, list):
                    return sorted(dynamic_frame_distribution_values)
            else:
                # Calculate the number of keyframes based on the total duration and linear_frames_per_keyframe
                return [i * linear_frame_distribution_value for i in range(len(images))]

        def extract_keyframe_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):
            if type_of_key_frame_influence == "dynamic":
                # Check if the input is a string or a list
                if isinstance(dynamic_key_frame_influence_values, str):
                    # Parse the dynamic key frame influence values without sorting
                    dynamic_values = [float(influence.strip()) for influence in dynamic_key_frame_influence_values.split(',')]
                elif isinstance(dynamic_key_frame_influence_values, list):
                    dynamic_values = dynamic_key_frame_influence_values
                else:
                    raise ValueError("Invalid type for dynamic_key_frame_influence_values. Must be string or list.")

                # Trim the dynamic_values to match the length of keyframe_positions
                return dynamic_values[:len(keyframe_positions)]

            else:
                # Create a list with the linear_key_frame_influence_value for each keyframe
                return [linear_key_frame_influence_value for _ in keyframe_positions]
        
        def extract_start_and_endpoint_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):
            if type_of_key_frame_influence == "dynamic":
                # If dynamic_key_frame_influence_values is a list of characters representing tuples, process it
                if isinstance(dynamic_key_frame_influence_values[0], str) and dynamic_key_frame_influence_values[0] == "(":
                    # Join the characters to form a single string and evaluate to convert into a list of tuples
                    string_representation = ''.join(dynamic_key_frame_influence_values)
                    dynamic_values = eval(f'[{string_representation}]')
                else:
                    # If it's already a list of tuples or a single tuple, use it directly
                    dynamic_values = dynamic_key_frame_influence_values if isinstance(dynamic_key_frame_influence_values, list) else [dynamic_key_frame_influence_values]
                return dynamic_values
            else:
                # Return a list of tuples with the linear_key_frame_influence_value as a tuple repeated for each position
                return [linear_key_frame_influence_value for _ in keyframe_positions]
                        
        def create_mask_batch(last_key_frame_position, weights, frames):
            # Hardcoded dimensions
            width, height = 512, 512

            # Calculate the reversed weights in a generalizable way (e.g., 0.6 becomes 0.4, 0.1 becomes 0.9)
            reversed_weights = [1.0 - weight for weight in weights]

            # Map frames to their corresponding reversed weights for easy lookup
            frame_to_weight = {frame: weights[i] for i, frame in enumerate(frames)}

            # Create masks for each frame up to last_key_frame_position
            masks = []
            for frame_number in range(last_key_frame_position):
                # Determine the strength of the mask
                strength = frame_to_weight.get(frame_number, 0.0)

                # Create the mask with the determined strength
                mask = torch.full((height, width), strength)
                masks.append(mask)

            # Convert list of masks to a single tensor
            masks_tensor = torch.stack(masks, dim=0)

            return masks_tensor
                                
        def adjust_influence_range(batch_index_from, batch_index_to_excl, last_key_frame_position, scale_factor, buffer):
            # Calculate the midpoint of the current range
            midpoint = (batch_index_from + batch_index_to_excl) // 2

            # Calculate the new range length
            new_range_length = int((batch_index_to_excl - batch_index_from) * scale_factor)

            # Adjusting both sides of the range
            if batch_index_from == 0:
                # Start is anchored at 0
                new_batch_index_from = 0
                new_batch_index_to_excl = batch_index_from + new_range_length
            elif batch_index_to_excl == last_key_frame_position:
                # End is anchored at last_key_frame_position
                new_batch_index_from = batch_index_to_excl - new_range_length
                new_batch_index_to_excl = last_key_frame_position
            else:
                # No anchoring, adjust both sides around the midpoint
                new_batch_index_from = midpoint - new_range_length // 2
                new_batch_index_to_excl = midpoint + new_range_length // 2

            # Remove minimum and maximum constraints

            return new_batch_index_from, new_batch_index_to_excl
                    
        def adjust_strength_values(strength_from, strength_to, multiplier):
            mid_point = (strength_from + strength_to) / 2
            range_half = abs(strength_to - strength_from) / 2

            # Adjust the range with the multiplier
            new_range_half = min(range_half * multiplier, 0.5)

            # Calculate new strength values, ensuring they stay within [0.0, 1.0]
            new_strength_from = max(mid_point - new_range_half, 0.0)
            new_strength_to = min(mid_point + new_range_half, 1.0)

            # Preserve the order of the original strength values
            if strength_from > strength_to:
                new_strength_from, new_strength_to = new_strength_to, new_strength_from

            return (new_strength_from, new_strength_to)
        
        def plot_weight_comparison(cn_frame_numbers, cn_weights, ipadapter_frame_numbers, ipadapter_weights, buffer):
            plt.figure(figsize=(12, 8))

            # Defining colors for each set of data
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Alternating the data sets with labels and colors
            max_length = max(len(cn_frame_numbers), len(ipadapter_frame_numbers))
            label_counter = 1 if buffer < 0 else 0  # Start from 1 if buffer < 0, else start from 0
            for i in range(max_length):
                # Label for cn_strength
                if i < len(cn_frame_numbers):
                    if i == 0 and buffer > 0:
                        label = 'cn_strength_buffer'
                    else:
                        label = f'cn_strength_{label_counter}'
                    plt.plot(cn_frame_numbers[i], cn_weights[i], marker='o', color=colors[i % len(colors)], label=label)

                # Label for ipa_strength
                if i < len(ipadapter_frame_numbers):
                    if i == 0 and buffer > 0:
                        label = 'ipa_strength_buffer'
                    else:
                        label = f'ipa_strength_{label_counter}'
                    plt.plot(ipadapter_frame_numbers[i], ipadapter_weights[i], marker='x', linestyle='--', color=colors[i % len(colors)], label=label)

                if label_counter == 0 or buffer < 0 or i > 0:
                    label_counter += 1

            plt.legend()
            max_weight = max([weight.max() for weight in cn_weights + ipadapter_weights]) * 1.5
            plt.ylim(0, max_weight)

            buffer_io = BytesIO()
            plt.savefig(buffer_io, format='png', bbox_inches='tight')
            plt.close()

            buffer_io.seek(0)
            img = Image.open(buffer_io)

            img_tensor = TT.ToTensor()(img)
        
            img_tensor = img_tensor.unsqueeze(0)
            
            img_tensor = img_tensor.permute([0, 2, 3, 1])

            return (img_tensor,)
                
        keyframe_positions = get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, images, linear_frame_distribution_value)                    
        cn_strength_values = extract_start_and_endpoint_values(type_of_cn_strength_distribution, dynamic_cn_strength_values, keyframe_positions, linear_cn_strength_value)                
        key_frame_influence_values = extract_keyframe_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value)                                                
        influence_ranges = calculate_dynamic_influence_ranges(keyframe_positions,key_frame_influence_values)        
        
        influence_ranges = add_starting_buffer(influence_ranges, buffer)                                    
        cn_strength_values = [literal_eval(val) if isinstance(val, str) else val for val in cn_strength_values]

        cn_frame_numbers = []
        cn_weights = []
        ipadapter_frame_numbers = []
        ipadapter_weights = []
        
        last_key_frame_position = (keyframe_positions[-1]) + buffer
        control_net = []
        for i, (start, end) in enumerate(influence_ranges):
            batch_index_from, batch_index_to_excl = influence_ranges[i]
            ipadapter_strength_multiplier = relative_ipadapter_strength
            ipadapter_influence_multiplier = relative_ipadapter_influence 

            if i == 0:
                if buffer > 0:  # First image with buffer
                    image = images[0]
                    strength_from = strength_to = cn_strength_values[0][1] if len(cn_strength_values) > 0 else (1.0, 1.0)
                    revert_direction_at_midpoint = False 
                    ipadapter_strength_multiplier = 1.0
                    ipadapter_influence_multiplier = 1.0
                    interpolation = "ease-in-out"
                else:
                    continue  # Skip first image without buffer
            elif i == 1: # First image
                image = images[0]
                strength_to, strength_from = cn_strength_values[0] if len(cn_strength_values) > 0 else (0.0, 1.0)
                revert_direction_at_midpoint = False 
                interpolation = "ease-in"

                
                
            elif i == len(images):  # Last image
                image = images[i-1]
                strength_from, strength_to = cn_strength_values[i-1] if i-1 < len(cn_strength_values) else (0.0, 1.0)
                revert_direction_at_midpoint = False     
                interpolation =  "ease-out"
                                
            else:  # Middle images
                image = images[i-1]
                strength_from, strength_to = cn_strength_values[i-1] if i-1 < len(cn_strength_values) else (0.0, 1.0)
                revert_direction_at_midpoint = True
                interpolation = "ease-in-out"    
                                
        
            latent_keyframe_interpolation_node = LatentKeyframeInterpolationNodeImport()
            weights, frame_numbers, latent_keyframe, = latent_keyframe_interpolation_node.load_keyframe(batch_index_from,strength_from,batch_index_to_excl,strength_to,interpolation,revert_direction_at_midpoint,last_key_frame_position,i,len(influence_ranges),buffer)                        

            cn_frame_numbers.append(frame_numbers)
            cn_weights.append(weights)

            scaled_soft_control_net_weights = ScaledSoftControlNetWeightsImport()
            control_net_weights, _ = scaled_soft_control_net_weights.load_weights(soft_scaled_cn_weights_multiplier,False)

            timestep_keyframe_node = TimestepKeyframeNodeImport()
            timestep_keyframe, = timestep_keyframe_node.load_keyframe(start_percent=0.0,control_net_weights=control_net_weights,t2i_adapter_weights=None,latent_keyframe=latent_keyframe,prev_timestep_keyframe=None)

            control_net_loader = ControlNetLoaderAdvancedImport()
            control_net, = control_net_loader.load_controlnet(control_net_name, timestep_keyframe)

            apply_advanced_control_net = AdvancedControlNetApplyImport()                                    
            positive, negative = apply_advanced_control_net.apply_controlnet(positive,negative,control_net,image.unsqueeze(0),1.0,0.0,1.0)       
                                     
            prepped_image, = prep_image(image=image.unsqueeze(0), interpolation="LANCZOS", crop_position="pad", sharpening=0.0)

            ipadapter_application = IPAdapterApply()
            
            ipa_strength_from, ipa_strength_to = adjust_strength_values(strength_from, strength_to, ipadapter_strength_multiplier)        
            
            ipa_batch_index_from, ipa_batch_index_to_excl = adjust_influence_range(batch_index_from, batch_index_to_excl, last_key_frame_position, ipadapter_influence_multiplier, buffer)

            ipa_weights, ipa_frame_numbers = calculate_weights(ipa_batch_index_from,ipa_batch_index_to_excl,ipa_strength_from, ipa_strength_to,interpolation, revert_direction_at_midpoint, last_key_frame_position,i, len(influence_ranges),buffer)
            

            ipadapter_frame_numbers.append(ipa_frame_numbers)
            ipadapter_weights.append(ipa_weights)


            masks = create_mask_batch(last_key_frame_position, weights, frame_numbers)
            
            model, = ipadapter_application.apply_ipadapter(ipadapter=ipadapter, model=model, weight=1.0, clip_vision=clip_vision, image=prepped_image, weight_type="original", noise=ipadapter_noise, embeds=None, attn_mask=masks, start_at=0.0, end_at=1.0, unfold_batch=True)        
        
        comparison_diagram, = plot_weight_comparison(cn_frame_numbers, cn_weights, ipadapter_frame_numbers, ipadapter_weights, buffer)
        
        return comparison_diagram, positive, negative, model

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Combined
    "BatchCreativeInterpolation": BatchCreativeInterpolationNode
    # "MaskGenerator": MaskGeneratorNode
    # "FILMVFIImport": FILMVFINode
    # Keyframes
    # "TimestepKeyframe": TimestepKeyframeNodeImport,
    # "LatentKeyframeImport": LatentKeyframeNodeImport,
    # "LatentKeyframeGroupImport": LatentKeyframeGroupImportNode,
    # "LatentKeyframeBatchedGroupImport": LatentKeyframeBatchedGroupNodeImport,
    # "LatentKeyframeTiming": LatentKeyframeInterpolationNodeImport,
    # Loaders
    # "ControlNetLoaderAdvancedImport": ControlNetLoaderAdvancedImport,
    # "DiffControlNetLoaderAdvancedImport": DiffControlNetLoaderAdvancedImport,
    # Conditioning
    # "ACN_AdvancedControlNetApplyImport": AdvancedControlNetApplyImport,
    # Weights
    # "ScaledSoftControlNetWeightsImport": ScaledSoftControlNetWeightsImport,
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
    # "MaskGenerator": "Mask Generator 🎞️🅟🅞🅜"
    # Keyframes
    # "TimestepKeyframe": "Timestep Keyframe 🎞️🅟🅞🅜",
    # "LatentKeyframe": "Latent Keyframe 🛂🅐🅒🅝",
    # "LatentKeyframeGroupImport": "Latent Keyframe Group 🛂🅐🅒🅝",
    # "LatentKeyframeBatchedGroup": "Latent Keyframe Batched Group 🛂🅐🅒🅝",
    # "LatentKeyframeTiming": "Latent Keyframe Interpolation 🛂🅐🅒🅝",
    # Loaders
    # "ControlNetLoaderAdvancedImport": "Load ControlNet Model (Advanced) 🛂🅐🅒🅝",
    # "DiffControlNetLoaderAdvancedImport": "Load ControlNet Model (diff Advanced) 🛂🅐🅒🅝",
    # Conditioning
    # "ACN_AdvancedControlNetApplyImport": "Apply Advanced ControlNet 🛂🅐🅒🅝",
    # Weights
    # "ScaledSoftControlNetWeightsImport": "Scaled Soft ControlNet Weights 🛂🅐🅒🅝",
    # "SoftControlNetWeights": "Soft ControlNet Weights 🛂🅐🅒🅝",
    # "CustomControlNetWeights": "Custom ControlNet Weights 🛂🅐🅒🅝",
    # "SoftT2IAdapterWeights": "Soft T2IAdapter Weights 🛂🅐🅒🅝",
    # "CustomT2IAdapterWeights": "Custom T2IAdapter Weights 🛂🅐🅒🅝",
    # Image
    # "LoadImagesFromDirectory": "Load Images [DEPRECATED] 🛂🅐🅒🅝"
}
