# Standard library imports
from ast import literal_eval
from io import BytesIO
import numpy as np
# Third-party library imports
import torch
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

# Local application/library specific imports
import folder_paths
from .imports.IPAdapterPlus import IPAdapterApplyImport, prep_image, IPAdapterEncoderImport
from .imports.AdvancedControlNet.latent_keyframe_nodes import LatentKeyframeInterpolationNodeImport
from .imports.AdvancedControlNet.weight_nodes import ScaledSoftUniversalWeightsImport
from .imports.AdvancedControlNet.nodes_sparsectrl import SparseIndexMethodNodeImport
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
                "linear_key_frame_influence_value": ("STRING", {"multiline": False, "default": "(1.0,1.0)"}),
                "dynamic_key_frame_influence_values": ("STRING", {"multiline": True, "default": "(1.0,1.0),(1.0,1.5)(1.0,0.5)"}),                
                "type_of_strength_distribution": (["linear", "dynamic"],),
                "linear_strength_value": ("STRING", {"multiline": False, "default": "(0.3,0.4)"}),
                "dynamic_strength_values": ("STRING", {"multiline": True, "default": "(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)"}),
                "soft_scaled_cn_weights_multiplier": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 10.0, "step": 0.1}),
                "buffer": ("INT", {"default": 4, "min": 0, "max": 16, "step": 1}), 
                "relative_cn_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "relative_ipadapter_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "ipadapter_noise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ipadapter_start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ipadapter_end_at": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cn_start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cn_end_at": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE","CONDITIONING","CONDITIONING","MODEL","SPARSE_METHOD","INT")
    # "comparison_diagram, positive, negative, model, sparse_indexes, last_key_frame_position"
    RETURN_NAMES = ("GRAPH","POSITIVE","NEGATIVE","MODEL","KEYFRAME_POSITIONS","BATCH_SIZE")
    FUNCTION = "combined_function"

    CATEGORY = "Steerable-Motion"

    def combined_function(self,positive,negative,images,model,ipadapter,clip_vision,control_net_name,
                          type_of_frame_distribution,linear_frame_distribution_value, dynamic_frame_distribution_values, 
                          type_of_key_frame_influence,linear_key_frame_influence_value,
                          dynamic_key_frame_influence_values,type_of_strength_distribution,
                          linear_strength_value,dynamic_strength_values, soft_scaled_cn_weights_multiplier,
                          buffer, relative_cn_strength,relative_ipadapter_strength,ipadapter_noise,
                          ipadapter_start_at=0.0,ipadapter_end_at=0.75, cn_start_at=0.0, cn_end_at=0.75):
        
        
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

        def plot_weight_comparison(cn_frame_numbers, cn_weights, ipadapter_frame_numbers, ipadapter_weights, buffer):
            plt.figure(figsize=(12, 8))

            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Handle None values for frame numbers and weights
            cn_frame_numbers = cn_frame_numbers if cn_frame_numbers is not None else []
            cn_weights = cn_weights if cn_weights is not None else []
            ipadapter_frame_numbers = ipadapter_frame_numbers if ipadapter_frame_numbers is not None else []
            ipadapter_weights = ipadapter_weights if ipadapter_weights is not None else []

            max_length = max(len(cn_frame_numbers), len(ipadapter_frame_numbers))
            label_counter = 1 if buffer < 0 else 0
            for i in range(max_length):
                if i < len(cn_frame_numbers):
                    label = 'cn_strength_buffer' if (i == 0 and buffer > 0) else f'cn_strength_{label_counter}'
                    plt.plot(cn_frame_numbers[i], cn_weights[i], marker='o', color=colors[i % len(colors)], label=label)

                if i < len(ipadapter_frame_numbers):
                    label = 'ipa_strength_buffer' if (i == 0 and buffer > 0) else f'ipa_strength_{label_counter}'
                    plt.plot(ipadapter_frame_numbers[i], ipadapter_weights[i], marker='x', linestyle='--', color=colors[i % len(colors)], label=label)

                if label_counter == 0 or buffer < 0 or i > 0:
                    label_counter += 1

            plt.legend()

            # Adjusted generator expression for max_weight
            all_weights = cn_weights + ipadapter_weights
            max_weight = max(max(sublist) for sublist in all_weights if sublist) * 1.5
            plt.ylim(0, max_weight)

            buffer_io = BytesIO()
            plt.savefig(buffer_io, format='png', bbox_inches='tight')
            plt.close()

            buffer_io.seek(0)
            img = Image.open(buffer_io)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.permute([0, 2, 3, 1])

            return img_tensor,                    
        
        def extract_strength_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):

            if type_of_key_frame_influence == "dynamic":
                # Process the dynamic_key_frame_influence_values depending on its format
                if isinstance(dynamic_key_frame_influence_values, str):
                    dynamic_values = eval(dynamic_key_frame_influence_values)
                else:
                    dynamic_values = dynamic_key_frame_influence_values

                # Iterate through the dynamic values and convert tuples with two values to three values
                dynamic_values_corrected = []
                for value in dynamic_values:
                    if len(value) == 2:
                        value = (value[0], value[1], value[0])
                    dynamic_values_corrected.append(value)

                return dynamic_values_corrected
            else:
                # Process for linear or other types
                if len(linear_key_frame_influence_value) == 2:
                    linear_key_frame_influence_value = (linear_key_frame_influence_value[0], linear_key_frame_influence_value[1], linear_key_frame_influence_value[0])
                return [linear_key_frame_influence_value for _ in range(len(keyframe_positions) - 1)]
                    
        def extract_influence_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):
            # Check and convert linear_key_frame_influence_value if it's a float or string float        
            # if it's a string that starts with a parenthesis, convert it to a tuple
            if isinstance(linear_key_frame_influence_value, str) and linear_key_frame_influence_value[0] == "(":
                linear_key_frame_influence_value = eval(linear_key_frame_influence_value)


            if not isinstance(linear_key_frame_influence_value, tuple):
                if isinstance(linear_key_frame_influence_value, (float, str)):
                    try:
                        value = float(linear_key_frame_influence_value)
                        linear_key_frame_influence_value = (value, value)
                    except ValueError:
                        raise ValueError("linear_key_frame_influence_value must be a float or a string representing a float")

            number_of_outputs = len(keyframe_positions) - 1

            if type_of_key_frame_influence == "dynamic":
                # Convert list of individual float values into tuples
                if all(isinstance(x, float) for x in dynamic_key_frame_influence_values):
                    dynamic_values = [(value, value) for value in dynamic_key_frame_influence_values]
                elif isinstance(dynamic_key_frame_influence_values[0], str) and dynamic_key_frame_influence_values[0] == "(":
                    string_representation = ''.join(dynamic_key_frame_influence_values)
                    dynamic_values = eval(f'[{string_representation}]')
                else:
                    dynamic_values = dynamic_key_frame_influence_values if isinstance(dynamic_key_frame_influence_values, list) else [dynamic_key_frame_influence_values]
                return dynamic_values[:number_of_outputs]
            else:
                return [linear_key_frame_influence_value for _ in range(number_of_outputs)]

        def calculate_weights(batch_index_from, batch_index_to, strength_from, strength_to, interpolation,revert_direction_at_midpoint, last_key_frame_position,i, number_of_items,buffer):

            # Initialize variables based on the position of the keyframe
            range_start = batch_index_from
            range_end = batch_index_to
            # if it's the first value, set influence range from 1.0 to 0.0
            if buffer > 0:
                if i == 0:
                    range_start = 0
                elif i == 1:
                    range_start = buffer
            else:
                if i == 1:
                    range_start = 0
            
            if i == number_of_items - 1:
                range_end = last_key_frame_position

            steps = range_end - range_start
            diff = strength_to - strength_from

            # Calculate index for interpolation
            index = np.linspace(0, 1, steps // 2 + 1) if revert_direction_at_midpoint else np.linspace(0, 1, steps)

            # Calculate weights based on interpolation type
            if interpolation == "linear":
                weights = np.linspace(strength_from, strength_to, len(index))
            elif interpolation == "ease-in":
                weights = diff * np.power(index, 2) + strength_from
            elif interpolation == "ease-out":
                weights = diff * (1 - np.power(1 - index, 2)) + strength_from
            elif interpolation == "ease-in-out":
                weights = diff * ((1 - np.cos(index * np.pi)) / 2) + strength_from
            
            if revert_direction_at_midpoint:
                weights = np.concatenate([weights, weights[::-1]])
                            
            # Generate frame numbers
            frame_numbers = np.arange(range_start, range_start + len(weights))

            # "Dropper" component: For keyframes with negative start, drop the weights
            if range_start < 0 and i > 0:
                drop_count = abs(range_start)
                weights = weights[drop_count:]
                frame_numbers = frame_numbers[drop_count:]

            # Dropper component: for keyframes a range_End is greater than last_key_frame_position, drop the weights
            if range_end > last_key_frame_position and i < number_of_items - 1:
                drop_count = range_end - last_key_frame_position
                weights = weights[:-drop_count]
                frame_numbers = frame_numbers[:-drop_count]

            return weights, frame_numbers       
        
        def process_weights(frame_numbers, weights, multiplier):
            # Multiply weights by the multiplier and apply the bounds of 0.0 and 1.0
            adjusted_weights = [min(max(weight * multiplier, 0.0), 1.0) for weight in weights]

            # Filter out frame numbers and weights where the weight is 0.0
            filtered_frames_and_weights = [(frame, weight) for frame, weight in zip(frame_numbers, adjusted_weights) if weight > 0.0]

            # Separate the filtered frame numbers and weights
            filtered_frame_numbers, filtered_weights = zip(*filtered_frames_and_weights) if filtered_frames_and_weights else ([], [])

            return list(filtered_frame_numbers), list(filtered_weights)
            
        def calculate_influence_frame_number(key_frame_position, next_key_frame_position, distance):
            # Calculate the absolute distance between key frames
            key_frame_distance = abs(next_key_frame_position - key_frame_position)
            
            # Apply the distance multiplier
            extended_distance = key_frame_distance * distance

            # Determine the direction of influence based on the positions of the key frames
            if key_frame_position < next_key_frame_position:
                # Normal case: influence extends forward
                influence_frame_number = key_frame_position + extended_distance
            else:
                # Reverse case: influence extends backward
                influence_frame_number = key_frame_position - extended_distance
            
            # Return the result rounded to the nearest integer
            return round(influence_frame_number)

        # GET KEYFRAME POSITIONS
        keyframe_positions = get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, images, linear_frame_distribution_value)                                            
        shifted_keyframes_position = [position + buffer - 2 for position in keyframe_positions]
        shifted_keyframe_positions_string = ','.join(str(pos) for pos in shifted_keyframes_position)        
        
        # GET SPARSE INDEXES
        sparseindexmethod = SparseIndexMethodNodeImport()        
        sparse_indexes, = sparseindexmethod.get_method(shifted_keyframe_positions_string)
        
        # ADD BUFFER TO KEYFRAME POSITIONS
        if buffer > 0:
            keyframe_positions = [position + buffer - 1 for position in keyframe_positions]
            keyframe_positions.insert(0, 0)
            
        # GET STRENGTH VALUES
        
        strength_values = extract_strength_values(type_of_strength_distribution, dynamic_strength_values, keyframe_positions, linear_strength_value)                        
        strength_values = [literal_eval(val) if isinstance(val, str) else val for val in strength_values]                
        corrected_strength_values = []
        for val in strength_values:
            if len(val) == 2:
                val = (val[0], val[1], val[0])
            corrected_strength_values.append(val)        
        strength_values = corrected_strength_values
                                    
        # GET KEYFRAME INFLUENCE VALUES
        key_frame_influence_values = extract_influence_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value)                
        key_frame_influence_values = [literal_eval(val) if isinstance(val, str) else val for val in key_frame_influence_values]        

        # CALCULATE LAST KEYFRAME POSITION                                                                                                                                           
        last_key_frame_position = (keyframe_positions[-1] + 1)
    
        # CREATE LISTS FOR WEIGHTS AND FRAME NUMBERS
        all_cn_frame_numbers = []
        all_cn_weights = []
        all_ipa_weights = []
        all_ipa_frame_numbers = []
        
        for i in range(len(keyframe_positions)):

            keyframe_position = keyframe_positions[i]                                    
            interpolation = "ease-in-out"
            # strength_from = strength_to = 1.0
                                        
            if i == 0: # buffer
                
                if buffer > 0:  # First image with buffer
                    image = images[0]
                    strength_from = strength_to = strength_values[0][1]                    
                else:
                    continue  # Skip first image without buffer
                batch_index_from = 0
                batch_index_to_excl = buffer
                weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, strength_from, strength_to, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                                    
            
            elif i == 1: # first image 

                # GET IMAGE AND KEYFRAME INFLUENCE VALUES              
                image = images[0]                
                key_frame_influence_from, key_frame_influence_to = key_frame_influence_values[0]                                
                start_strength, mid_strength, end_strength = strength_values[0]
                                
                keyframe_position = keyframe_positions[i]
                next_key_frame_position = keyframe_positions[i+1]
                
                batch_index_from = keyframe_position                
                batch_index_to_excl = calculate_influence_frame_number(keyframe_position, next_key_frame_position, key_frame_influence_to)
                weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, mid_strength, end_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                                    
                # interpolation = "ease-in"                                
            
            elif i == len(images):  # last image

                # GET IMAGE AND KEYFRAME INFLUENCE VALUES
                image = images[i-1]
                key_frame_influence_from,key_frame_influence_to = key_frame_influence_values[i-1]       
                start_strength, mid_strength, end_strength = strength_values[i-1]
                # strength_from, strength_to = cn_strength_values[i-1]

                keyframe_position = keyframe_positions[i]
                previous_key_frame_position = keyframe_positions[i-1]

                batch_index_from = calculate_influence_frame_number(keyframe_position, previous_key_frame_position, key_frame_influence_from)
                batch_index_to_excl = keyframe_position
                weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, start_strength, mid_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                                    
                # interpolation =  "ease-out"                                
            
            else:  # middle images

                # GET IMAGE AND KEYFRAME INFLUENCE VALUES
                image = images[i-1]   
                key_frame_influence_from,key_frame_influence_to = key_frame_influence_values[i-1]             
                start_strength, mid_strength, end_strength = strength_values[i-1]
                keyframe_position = keyframe_positions[i]
                           
                # CALCULATE WEIGHTS FOR FIRST HALF
                previous_key_frame_position = keyframe_positions[i-1]   
                batch_index_from = calculate_influence_frame_number(keyframe_position, previous_key_frame_position, key_frame_influence_from)                
                batch_index_to_excl = keyframe_position                
                first_half_weights, first_half_frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, start_strength, mid_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                
                
                # CALCULATE WEIGHTS FOR SECOND HALF                
                next_key_frame_position = keyframe_positions[i+1]
                batch_index_from = keyframe_position
                batch_index_to_excl = calculate_influence_frame_number(keyframe_position, next_key_frame_position, key_frame_influence_to)                                
                second_half_weights, second_half_frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, mid_strength, end_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)
                
                # COMBINE FIRST AND SECOND HALF
                weights = np.concatenate([first_half_weights, second_half_weights])                
                frame_numbers = np.concatenate([first_half_frame_numbers, second_half_frame_numbers])
                    
            # IMPORT REQUIRED NODES
            latent_keyframe_interpolation_node = LatentKeyframeInterpolationNodeImport()
            scaled_soft_control_net_weights = ScaledSoftUniversalWeightsImport()
            timestep_keyframe_node = TimestepKeyframeNodeImport()
            control_net_loader = ControlNetLoaderAdvancedImport()
            apply_advanced_control_net = AdvancedControlNetApplyImport()
            ipadapter_application = IPAdapterApplyImport()
            ipadapter_encoder = IPAdapterEncoderImport()
                                                                                                             
            # IF CONTROL NET IS USED, APPLY IT
            if relative_cn_strength > 0.0:
                cn_frame_numbers, cn_weights = process_weights(frame_numbers, weights, relative_cn_strength)        
                latent_keyframe, = latent_keyframe_interpolation_node.load_keyframe(cn_weights, cn_frame_numbers)
                control_net_weights, _ = scaled_soft_control_net_weights.load_weights(soft_scaled_cn_weights_multiplier, False)
                timestep_keyframe = timestep_keyframe_node.load_keyframe(start_percent=0.0, control_net_weights=control_net_weights, latent_keyframe=latent_keyframe, prev_timestep_keyframe=None)[0]            
                control_net = control_net_loader.load_controlnet(control_net_name, timestep_keyframe)[0]
                positive, negative = apply_advanced_control_net.apply_controlnet(positive, negative, control_net, image.unsqueeze(0), 1.0, cn_start_at, cn_end_at)
                all_cn_frame_numbers.append(cn_frame_numbers)
                all_cn_weights.append(cn_weights)
            else:
                all_cn_frame_numbers = None
                all_cn_weights = None
            
            # IF IP ADAPTER IS USED, APPLY IT
            if relative_ipadapter_strength > 0.0:                    
                ipa_frame_numbers, ipa_weights = process_weights(frame_numbers, weights, relative_ipadapter_strength)     
                prepped_image = prep_image(image=image.unsqueeze(0), interpolation="LANCZOS", crop_position="pad", sharpening=0.0)[0]                        
                mask = create_mask_batch(last_key_frame_position, ipa_weights, ipa_frame_numbers)                        
                embed, = ipadapter_encoder.preprocess(clip_vision, prepped_image, True, 0.0, 1.0)                        
                model, = ipadapter_application.apply_ipadapter(ipadapter=ipadapter, model=model, weight=1.0, image=None, weight_type="original", 
                                                  noise=ipadapter_noise, embeds=embed, attn_mask=mask, start_at=ipadapter_start_at, end_at=ipadapter_end_at, unfold_batch=True)    
                all_ipa_frame_numbers.append(ipa_frame_numbers)
                all_ipa_weights.append(ipa_weights)
            else:
                all_ipa_frame_numbers = None
                all_ipa_weights = None
        
        comparison_diagram, = plot_weight_comparison(all_cn_frame_numbers, all_cn_weights, all_ipa_frame_numbers, all_ipa_weights, buffer)

        return comparison_diagram, positive, negative, model, sparse_indexes, last_key_frame_position

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "BatchCreativeInterpolation": BatchCreativeInterpolationNode    
}

NODE_DISPLAY_NAME_MAPPINGS = {    
    "BatchCreativeInterpolation": "Batch Creative Interpolation 🎞️🅢🅜"
}
