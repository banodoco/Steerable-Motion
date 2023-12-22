# Standard library imports
from ast import literal_eval
from io import BytesIO

# Third-party library imports
import torch
import torchvision.transforms as TT
from PIL import Image
import matplotlib.pyplot as plt

# Local application/library specific imports
import folder_paths
from .imports.IPAdapterPlus import (IPAdapterApplyImport, prep_image, IPAdapterEncoderImport,)
from .imports.AdvancedControlNet.latent_keyframe_nodes import (
    calculate_weights,
    LatentKeyframeInterpolationNodeImport
)
from .imports.AdvancedControlNet.weight_nodes import ScaledSoftUniversalWeightsImport
from .imports.AdvancedControlNet.nodes import ControlNetLoaderAdvancedImport, AdvancedControlNetApplyImport,TimestepKeyframeNodeImport
    
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

            # Alternating the datasets by assigning unique labels and colors for plotting
            max_length = max(len(cn_frame_numbers), len(ipadapter_frame_numbers))
            # Determine starting label index based on the buffer value
            label_index = 1 if buffer < 0 else 0

            for i in range(max_length):
                # Assign label and plot data for cn_strength
                if i < len(cn_frame_numbers):
                    # Special label for the first item when buffer is positive
                    if i == 0 and buffer > 0:
                        label = 'cn_strength_buffer'
                    else:
                        label = f'cn_strength_{label_index}'

                    # Plotting the data point with a unique color and label
                    color_index = i % len(colors)
                    plt.plot(cn_frame_numbers[i], cn_weights[i], marker='o', color=colors[color_index], label=label)

                # Define a function to generate labels and plot data points
                def plot_data_with_label(frame_numbers, weights, label_base, marker, linestyle, buffer, colors):
                    label_index = 1 if buffer < 0 else 0

                    for i in range(len(frame_numbers)):
                        # Special label for the first item when buffer is positive
                        label = f'{label_base}_buffer' if i == 0 and buffer > 0 else f'{label_base}_{label_index}'

                        # Plotting the data point with a unique color and label
                        plt.plot(frame_numbers[i], weights[i], marker=marker, linestyle=linestyle,
                                 color=colors[i % len(colors)], label=label)

                        # Increment label index if not the first item or buffer is negative
                        if label_index == 0 or buffer < 0 or i > 0:
                            label_index += 1

                # Usage of the function
                plot_data_with_label(ipadapter_frame_numbers, ipadapter_weights, 'ipa_strength', 'x', '--', buffer,
                                     colors)

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
        cn_frame_numbers, cn_weights, ipadapter_frame_numbers, ipadapter_weights = [], [], [], []        
        last_key_frame_position = (keyframe_positions[-1]) + buffer        

        
        embeds = []
        masks = []
        existing_embeds = []

        for i, (start, end) in enumerate(influence_ranges):
            # set basic values
            batch_index_from, batch_index_to_excl = influence_ranges[i]
            ipadapter_strength_multiplier = relative_ipadapter_strength
            ipadapter_influence_multiplier = relative_ipadapter_influence 

            # Default values
            revert_direction_at_midpoint = False
            interpolation = "ease-in-out"
            strength_from = strength_to = 1.0

            if i == 0:
                if buffer > 0:  # First image with buffer
                    image = images[0]
                    strength_from = strength_to = cn_strength_values[0][1] if len(cn_strength_values) > 0 else (1.0, 1.0)                                    
                    ipadapter_influence_multiplier = 1.0
                    interpolation = "ease-in-out"
                else:
                    continue  # Skip first image without buffer
            elif i == 1: # First image
                image = images[0]
                strength_to, strength_from = cn_strength_values[0] if len(cn_strength_values) > 0 else (0.0, 1.0)                
                interpolation = "ease-in"                                
            elif i == len(images):  # Last image
                image = images[i-1]
                strength_from, strength_to = cn_strength_values[i-1] if i-1 < len(cn_strength_values) else (0.0, 1.0)                 
                interpolation =  "ease-out"                                
            else:  # Middle images
                image = images[i-1]
                strength_from, strength_to = cn_strength_values[i-1] if i-1 < len(cn_strength_values) else (0.0, 1.0)                                                                
                revert_direction_at_midpoint = True
        
            # Import necessary modules
            latent_keyframe_interpolation_node = LatentKeyframeInterpolationNodeImport()
            scaled_soft_control_net_weights = ScaledSoftUniversalWeightsImport()
            timestep_keyframe_node = TimestepKeyframeNodeImport()
            control_net_loader = ControlNetLoaderAdvancedImport()
            apply_advanced_control_net = AdvancedControlNetApplyImport()
            ipadapter_application = IPAdapterApplyImport()
            ipadapter_encoder = IPAdapterEncoderImport()
            # ipadapter_batcher = IPAdapterBatchEmbedsImport()

            # Load keyframe and append frame numbers and weights
            weights, frame_numbers, latent_keyframe = latent_keyframe_interpolation_node.load_keyframe(
                batch_index_from, strength_from, batch_index_to_excl, strength_to, interpolation, revert_direction_at_midpoint, last_key_frame_position, i, len(influence_ranges), buffer)
            cn_frame_numbers.append(frame_numbers)
            cn_weights.append(weights)

            # Load weights and keyframe
            control_net_weights, _ = scaled_soft_control_net_weights.load_weights(soft_scaled_cn_weights_multiplier, False)
            timestep_keyframe = timestep_keyframe_node.load_keyframe(start_percent=0.0, control_net_weights=control_net_weights, latent_keyframe=latent_keyframe, prev_timestep_keyframe=None)[0]

            # Load and apply control net
            control_net = control_net_loader.load_controlnet(control_net_name, timestep_keyframe)[0]
            positive, negative = apply_advanced_control_net.apply_controlnet(positive, negative, control_net, image.unsqueeze(0), 1.0, 0.0, 1.0)

            # Prepare image
            prepped_image = prep_image(image=image.unsqueeze(0), interpolation="LANCZOS", crop_position="pad", sharpening=0.0)[0]

            # Adjust strength values and influence range    
            ipa_strength_from, ipa_strength_to = adjust_strength_values(strength_from, strength_to, ipadapter_strength_multiplier)            
            ipa_batch_index_from, ipa_batch_index_to_excl = adjust_influence_range(batch_index_from, batch_index_to_excl, last_key_frame_position, ipadapter_influence_multiplier, buffer)

            # Calculate weights and append frame numbers and weights
            ipa_weights, ipa_frame_numbers = calculate_weights(ipa_batch_index_from, ipa_batch_index_to_excl, ipa_strength_from, ipa_strength_to, interpolation, revert_direction_at_midpoint, last_key_frame_position, i, len(influence_ranges), buffer)
            ipadapter_frame_numbers.append(ipa_frame_numbers)
            ipadapter_weights.append(ipa_weights)

                        
            mask = create_mask_batch(last_key_frame_position, ipa_weights, frame_numbers)        
            # add mask to masks list
            masks.append(mask)
            
            embed, = ipadapter_encoder.preprocess(clip_vision, prepped_image, True, 0.0, 1.0)
            # add embeds to current batch
            embeds.append(embed)    

            model, = ipadapter_application.apply_ipadapter(ipadapter=ipadapter, model=model, weight=1.0, image=None, weight_type="original", 
                                                noise=ipadapter_noise, embeds=embed, attn_mask=mask, start_at=0.0, end_at=1.0, unfold_batch=True)
                                    

        # print out the format for the embeds        
                        
        # merged_embeds = torch.cat(embeds, dim=1)

        # stacked_masks = torch.stack(masks)

        # merged_masks = torch.cat(masks, dim=1)
        


        comparison_diagram, = plot_weight_comparison(cn_frame_numbers, cn_weights, ipadapter_frame_numbers, ipadapter_weights, buffer)
        
        return comparison_diagram, positive, negative, model


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "BatchCreativeInterpolation": BatchCreativeInterpolationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {    
    "BatchCreativeInterpolation": "Batch Creative Interpolation 🎞️🅢🅜"
}
