# Standard library imports
from ast import literal_eval
from io import BytesIO
import logging
import math
import gc

# Third-party library imports
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Local application/library specific imports
from .imports.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterBatchImport, IPAdapterTiledBatchImport, IPAdapterTiledImport, PrepImageForClipVisionImport, IPAdapterAdvancedImport, IPAdapterNoiseImport
from .imports.ComfyUI_Frame_Interpolation.vfi_models.film import FILM_VFIImport
from comfy.utils import common_upscale

try:
    from .utils import log # If your .utils has a log object
except ImportError:
    log = logging.getLogger(__name__) # Fallback to standard logging

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
                "type_of_frame_distribution": (["linear", "dynamic"],),
                "linear_frame_distribution_value": ("INT", {"default": 16, "min": 4, "max": 64, "step": 1}),     
                "dynamic_frame_distribution_values": ("STRING", {"multiline": True, "default": "0,10,26,40"}),                
                "type_of_key_frame_influence": (["linear", "dynamic"],),
                "linear_key_frame_influence_value": ("STRING", {"multiline": False, "default": "(1.0,1.0)"}),
                "dynamic_key_frame_influence_values": ("STRING", {"multiline": True, "default": "(1.0,1.0),(1.0,1.5)(1.0,0.5)"}),                
                "type_of_strength_distribution": (["linear", "dynamic"],),
                "linear_strength_value": ("STRING", {"multiline": False, "default": "(0.3,0.4)"}),
                "dynamic_strength_values": ("STRING", {"multiline": True, "default": "(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)"}),                                                                                                                                            
                "buffer": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),       
                "high_detail_mode": ("BOOLEAN", {"default": True}),                                                                                     
            },
            "optional": {
                "base_ipa_advanced_settings": ("ADVANCED_IPA_SETTINGS",),
                "detail_ipa_advanced_settings": ("ADVANCED_IPA_SETTINGS",),
            }
        }

    RETURN_TYPES = ("IMAGE","CONDITIONING","CONDITIONING","MODEL","STRING","INT", "INT", "STRING")
    RETURN_NAMES = ("GRAPH","POSITIVE","NEGATIVE","MODEL","KEYFRAME_POSITIONS","BATCH_SIZE", "BUFFER","FRAMES_TO_DROP")
    FUNCTION = "combined_function"

    CATEGORY = "Steerable-Motion"

    def combined_function(self,positive,negative,images,model,ipadapter,clip_vision,
                          type_of_frame_distribution,linear_frame_distribution_value, 
                          dynamic_frame_distribution_values, type_of_key_frame_influence,linear_key_frame_influence_value,
                          dynamic_key_frame_influence_values,type_of_strength_distribution,
                          linear_strength_value,dynamic_strength_values,
                          buffer, high_detail_mode,base_ipa_advanced_settings=None,
                          detail_ipa_advanced_settings=None):
        # set the matplotlib backend to 'Agg' to prevent crash on macOS
        # 'Agg' is a non-interactive backend that can be used in a non-main thread
        matplotlib.use('Agg')

        def get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, images, linear_frame_distribution_value):
            if type_of_frame_distribution == "dynamic":
                # Check if the input is a string or a list
                if isinstance(dynamic_frame_distribution_values, str):
                    # Parse the keyframe positions, sort them, and then increase each by 1 except the first
                    keyframes = sorted([int(kf.strip()) for kf in dynamic_frame_distribution_values.split(',')])
                elif isinstance(dynamic_frame_distribution_values, list):
                    # Sort the list and then increase each by 1 except the first
                    keyframes = sorted(dynamic_frame_distribution_values)
            else:
                # Calculate the number of keyframes based on the total duration and linear_frames_per_keyframe
                # Increase each by 1 except the first
                keyframes = [(i * linear_frame_distribution_value) for i in range(len(images))]

            # Increase all values by 1 except the first
            if len(keyframes) > 1:
                return [keyframes[0]] + [kf + 1 for kf in keyframes[1:]]
            else:
                return keyframes
        
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
        
        def create_weight_batch(last_key_frame_position, weights, frames):

            # Map frames to their corresponding reversed weights for easy lookup
            frame_to_weight = {frame: weights[i] for i, frame in enumerate(frames)}

            # Create weights for each frame up to last_key_frame_position
            weights = []
            for frame_number in range(last_key_frame_position):
                # Determine the strength of the weight
                strength = frame_to_weight.get(frame_number, 0.0)

                weights.append(strength)

            return weights

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
        
        if buffer > 0:
            # add front buffer
            keyframe_positions = [position + buffer - 1 for position in keyframe_positions]
            keyframe_positions.insert(0, 0)
            # add end buffer
            last_position_with_buffer = keyframe_positions[-1] + buffer + 1
            keyframe_positions.append(last_position_with_buffer)


        # GET BASE ADVANCED SETTINGS OR SET DEFAULTS
        if base_ipa_advanced_settings is None:
            if high_detail_mode:
                base_ipa_advanced_settings = {
                    "ipa_starts_at": 0.0,
                    "ipa_ends_at": 0.3,
                    "ipa_weight_type": "ease in-out",
                    "ipa_weight": 1.0,
                    "ipa_embeds_scaling": "V only",
                    "ipa_noise_strength": 0.0,                    
                    "use_image_for_noise": False,
                    "type_of_noise": "fade",
                    "noise_blur": 0,
                }
            else:
                base_ipa_advanced_settings = {
                    "ipa_starts_at": 0.0,
                    "ipa_ends_at": 0.75,
                    "ipa_weight_type": "ease in-out",
                    "ipa_weight": 1.0,
                    "ipa_embeds_scaling": "V only",
                    "ipa_noise_strength": 0.0,
                    "use_image_for_noise": False,
                    "type_of_noise": "fade",
                    "noise_blur": 0,
                }
                                
        # GET DETAILED ADVANCED SETTINGS OR SET DEFAULTS
        if detail_ipa_advanced_settings is None:
            if high_detail_mode:
                detail_ipa_advanced_settings = {
                    "ipa_starts_at": 0.25,
                    "ipa_ends_at": 0.75,
                    "ipa_weight_type": "ease in-out",
                    "ipa_weight": 1.0,
                    "ipa_embeds_scaling": "V only",
                    "ipa_noise_strength": 0.0,
                    "use_image_for_noise": False,
                    "type_of_noise": "fade",
                    "noise_blur": 0,
                }                 
        
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
        if len(keyframe_positions) == 4:
            last_key_frame_position = (keyframe_positions[-1]) - 1
        else:                                                                                                                                        
            last_key_frame_position = (keyframe_positions[-1])
    
        class IPBin:
            def __init__(self):
                self.indicies = []
                self.image_schedule = []
                self.weight_schedule = []
                self.imageBatch = []
                self.bigImageBatch = []
                self.noiseBatch = []
                self.bigNoiseBatch = []

            def length(self):
                return len(self.image_schedule)
            
            def add(self, image, big_image, noise, big_noise, image_index, frame_numbers, weights):
                # Map frames to their corresponding reversed weights for easy lookup
                frame_to_weight = {frame: weights[i] for i, frame in enumerate(frame_numbers)}
                # Search for image index, if it isn't there add the image
                try:
                    index = self.indicies.index(image_index)
                except ValueError:
                    self.imageBatch.append(image)
                    self.bigImageBatch.append(big_image)
                    if noise is not None: self.noiseBatch.append(noise) 
                    if big_noise is not None: self.bigNoiseBatch.append(big_noise)
                    self.indicies.append(image_index)
                    index = self.indicies.index(image_index)
                
                self.image_schedule.extend([index] * (frame_numbers[-1] + 1 - len(self.image_schedule)))
                self.weight_schedule.extend([0] * (frame_numbers[0] - len(self.weight_schedule)))
                self.weight_schedule.extend(frame_to_weight[frame] for frame in range(frame_numbers[0], frame_numbers[-1] + 1))

        # CREATE LISTS FOR WEIGHTS AND FRAME NUMBERS
        all_cn_frame_numbers = []
        all_cn_weights = []
        all_ipa_weights = []
        all_ipa_frame_numbers = []
        # Start with one bin
        bins = [IPBin()]
        
        for i in range(len(keyframe_positions)):
            
            keyframe_position = keyframe_positions[i]                                    
            interpolation = "ease-in-out"
            # strength_from = strength_to = 1.0
            image_index = 0    
            if i == 0: # buffer                
                
                image = images[0]
                image_index = 0
                strength_from = strength_to = strength_values[0][1]                    

                batch_index_from = 0
                batch_index_to_excl = buffer
                weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, strength_from, strength_to, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                                    
            
            elif i == 1: # first image 

                # GET IMAGE AND KEYFRAME INFLUENCE VALUES              
                image = images[i-1]                
                image_index = i-1
                key_frame_influence_from, key_frame_influence_to = key_frame_influence_values[i-1]
                start_strength, mid_strength, end_strength = strength_values[i-1]
                                
                keyframe_position = keyframe_positions[i] + 1
                next_key_frame_position = keyframe_positions[i+1] + 1
                                                
                batch_index_from = keyframe_position                
                batch_index_to_excl = calculate_influence_frame_number(keyframe_position, next_key_frame_position, key_frame_influence_to)                
                weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, mid_strength, end_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                                    
                # interpolation = "ease-in"                                
            
            elif i == len(keyframe_positions) - 2: # last image

                # GET IMAGE AND KEYFRAME INFLUENCE VALUES
                image = images[i-1]
                image_index = i - 1
                key_frame_influence_from,key_frame_influence_to = key_frame_influence_values[i-1]       
                start_strength, mid_strength, end_strength = strength_values[i-1]
                if len(keyframe_positions) == 4:
                    keyframe_position = keyframe_positions[i] - 1
                else:                    
                    keyframe_position = keyframe_positions[i]

                previous_key_frame_position = keyframe_positions[i-1]
                                                
                batch_index_from = calculate_influence_frame_number(keyframe_position, previous_key_frame_position, key_frame_influence_from)
                
                batch_index_to_excl = keyframe_position + 1
                weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, start_strength, mid_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                                    
                # interpolation =  "ease-out"    

            elif i == len(keyframe_positions) - 1: # buffer

                image = images[i-2]
                image_index = i - 2
                strength_from = strength_to = strength_values[i-2][1]

                if len(keyframe_positions) == 4:
                    batch_index_from = keyframe_positions[i-1]
                    batch_index_to_excl = last_key_frame_position - 1
                else:
                    batch_index_from = keyframe_positions[i-1] + 1
                    batch_index_to_excl = last_key_frame_position

                weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, strength_from, strength_to, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)
            
            else:  # middle images

                # GET IMAGE AND KEYFRAME INFLUENCE VALUES
                image = images[i-1]
                image_index = i - 1   
                key_frame_influence_from,key_frame_influence_to = key_frame_influence_values[i-1]             
                start_strength, mid_strength, end_strength = strength_values[i-1]
                keyframe_position = keyframe_positions[i]
                           
                # CALCULATE WEIGHTS FOR FIRST HALF
                previous_key_frame_position = keyframe_positions[i-1]   
                batch_index_from = calculate_influence_frame_number(keyframe_position, previous_key_frame_position, key_frame_influence_from)                
                batch_index_to_excl = keyframe_position + 1
                first_half_weights, first_half_frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, start_strength, mid_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                
                
                # CALCULATE WEIGHTS FOR SECOND HALF                
                next_key_frame_position = keyframe_positions[i+1]
                batch_index_from = keyframe_position
                batch_index_to_excl = calculate_influence_frame_number(keyframe_position, next_key_frame_position, key_frame_influence_to) + 2
                second_half_weights, second_half_frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, mid_strength, end_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)
                
                # COMBINE FIRST AND SECOND HALF
                weights = np.concatenate([first_half_weights, second_half_weights])                
                frame_numbers = np.concatenate([first_half_frame_numbers, second_half_frame_numbers])
                                                                                                                                                                                                                   
            # PROCESS WEIGHTS
            ipa_frame_numbers, ipa_weights = process_weights(frame_numbers, weights, 1.0)    

  
            prepare_for_clip_vision = PrepImageForClipVisionImport()
            prepped_image, = prepare_for_clip_vision.prep_image(image=image.unsqueeze(0), interpolation="LANCZOS", crop_position="pad", sharpening=0.1)
            
            if base_ipa_advanced_settings["ipa_noise_strength"] > 0:
                if base_ipa_advanced_settings["use_image_for_noise"]:
                    noise_image = prepped_image
                else:
                    noise_image = None
                ipa_noise = IPAdapterNoiseImport()
                negative_noise, = ipa_noise.make_noise(type=base_ipa_advanced_settings["type_of_noise"], strength=base_ipa_advanced_settings["ipa_noise_strength"], blur=base_ipa_advanced_settings["noise_blur"], image_optional=noise_image)
            else:
                negative_noise = None

            if high_detail_mode and detail_ipa_advanced_settings["ipa_noise_strength"] > 0:
                if detail_ipa_advanced_settings["use_image_for_noise"]:
                    noise_image = image.unsqueeze(0)
                else:
                    noise_image = None
                ipa_noise = IPAdapterNoiseImport()
                big_negative_noise, = ipa_noise.make_noise(type=detail_ipa_advanced_settings["type_of_noise"], strength=detail_ipa_advanced_settings["ipa_noise_strength"], blur=detail_ipa_advanced_settings["noise_blur"], image_optional=noise_image)                    
            else:
                big_negative_noise = None

            if len(ipa_frame_numbers) > 0:
                # Fill up bins with image frames. Bins will automatically be created when needed but all the frames should be able to be packed into two bins
                active_index = -1
                # Find a bin that we can fit the next image into
                for i, bin in enumerate(bins):
                    if bin.length() <= ipa_frame_numbers[0]:
                        active_index = i
                        break
                # If we didn't find a suitable bin, add a new one
                if active_index == -1:
                    bins.append(IPBin())
                    active_index = len(bins) - 1

                # Add the image to the bin
                bins[active_index].add(prepped_image, image.unsqueeze(0), negative_noise, big_negative_noise, image_index, ipa_frame_numbers, ipa_weights)
  
            all_ipa_frame_numbers.append(ipa_frame_numbers)
            all_ipa_weights.append(ipa_weights)
        
        # Go through the bins and create IPAdapters for them
        for i, bin in enumerate(bins):
            ipadapter_application = IPAdapterBatchImport()
            negative_noise = torch.cat(bin.noiseBatch, dim=0) if len(bin.noiseBatch) > 0 else None
            model, *_ = ipadapter_application.apply_ipadapter(model=model, ipadapter=ipadapter, image=torch.cat(bin.imageBatch, dim=0), weight=[x * base_ipa_advanced_settings["ipa_weight"] for x in bin.weight_schedule], weight_type=base_ipa_advanced_settings["ipa_weight_type"], start_at=base_ipa_advanced_settings["ipa_starts_at"], end_at=base_ipa_advanced_settings["ipa_ends_at"], clip_vision=clip_vision,image_negative=negative_noise,embeds_scaling=base_ipa_advanced_settings["ipa_embeds_scaling"], encode_batch_size=1, image_schedule=bin.image_schedule)                
            if high_detail_mode:
                tiled_ipa_application = IPAdapterTiledBatchImport()
                negative_noise = torch.cat(bin.bigNoiseBatch, dim=0) if len(bin.bigNoiseBatch) > 0 else None
                model, *_ = tiled_ipa_application.apply_tiled(model=model, ipadapter=ipadapter, image=torch.cat(bin.bigImageBatch, dim=0), weight=[x * detail_ipa_advanced_settings["ipa_weight"] for x in bin.weight_schedule], weight_type=detail_ipa_advanced_settings["ipa_weight_type"], start_at=detail_ipa_advanced_settings["ipa_starts_at"], end_at=detail_ipa_advanced_settings["ipa_ends_at"], clip_vision=clip_vision,sharpening=0.1,image_negative=negative_noise,embeds_scaling=detail_ipa_advanced_settings["ipa_embeds_scaling"], encode_batch_size=1, image_schedule=bin.image_schedule)

        comparison_diagram, = plot_weight_comparison(all_cn_frame_numbers, all_cn_weights, all_ipa_frame_numbers, all_ipa_weights, buffer)
        return comparison_diagram, positive, negative, model, shifted_keyframe_positions_string, last_key_frame_position, buffer, shifted_keyframes_position

class RemoveAndInterpolateFramesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "frames_to_drop": ("STRING", {"multiline": True, "default": "[8, 16, 24]"}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "replace_and_interpolate_frames"
    CATEGORY = "Steerable-Motion"

    def replace_and_interpolate_frames(self, images: torch.Tensor, frames_to_drop: str):
        if isinstance(frames_to_drop, str):
            frames_to_drop = eval(frames_to_drop)

        frames_to_drop = sorted(frames_to_drop, reverse=True)
        
        # Create instance of FILM_VFI within the function
        film_vfi = FILM_VFIImport()  # Assuming FILM_VFI does not require any special setup

        for index in frames_to_drop:
            if 0 < index < images.shape[0] - 1:
                # Extract the two surrounding frames
                batch = images[index-1:index+2:2]

                # Process through FILM_VFI
                interpolated_frames = film_vfi.vfi(
                    ckpt_name='film_net_fp32.pt', 
                    frames=batch, 
                    clear_cache_after_n_frames=10, 
                    multiplier=2
                )[0]  # Assuming vfi returns a tuple and the first element is the interpolated frames

                # Replace the original frames at the location
                images = torch.cat((images[:index-1], interpolated_frames, images[index+2:]))

        return (images,)
        

class IpaConfigurationNode:
    WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle']
    IPA_EMBEDS_SCALING_OPTIONS = ["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipa_starts_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ipa_ends_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ipa_weight_type": (cls.WEIGHT_TYPES,),
                "ipa_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "ipa_embeds_scaling": (cls.IPA_EMBEDS_SCALING_OPTIONS,),
                "ipa_noise_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_image_for_noise": ("BOOLEAN", {"default": False}),                                  
                "type_of_noise": (["fade", "dissolve", "gaussian", "shuffle"], ),
                "noise_blur": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1 }),
            },
            "optional": {}
        }
    
    FUNCTION = "process_inputs"
    RETURN_TYPES = ("ADVANCED_IPA_SETTINGS",)
    RETURN_NAMES = ("configuration",)
    CATEGORY = "Steerable-Motion"

    @classmethod
    def process_inputs(cls, ipa_starts_at, ipa_ends_at, ipa_weight_type, ipa_weight, ipa_embeds_scaling, ipa_noise_strength, use_image_for_noise, type_of_noise, noise_blur):
        return {
            "ipa_starts_at": ipa_starts_at,
            "ipa_ends_at": ipa_ends_at,
            "ipa_weight_type": ipa_weight_type,
            "ipa_weight": ipa_weight,
            "ipa_embeds_scaling": ipa_embeds_scaling,
            "ipa_noise_strength": ipa_noise_strength,
            "use_image_for_noise": use_image_for_noise,
            "type_of_noise": type_of_noise,
            "noise_blur": noise_blur,
        },

class VideoFrameExtractorAndMaskGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video_frames": ("IMAGE", {"tooltip": "Input video frames (IMAGE batch) to extract from."}),
                "total_output_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Total number of frames for the output guidance video and masks. Must satisfy: (frames - 1) divisible by 4."}),
                "frame_selection_string": ("STRING", {"default": "0, 10:20", "multiline": False, "tooltip": "Comma-separated integers or ranges (e.g., 0, 5, 10:15, 20) of frames to extract from input video. Takes precedence over depth_frames."}),
                "empty_frame_fill_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Grayscale level (0.0 black, 1.0 white) for frames not explicitly selected or filled by depth."}),
            },
            "optional": {
                "depth_video_frames": ("IMAGE", {"tooltip": "Optional depth frames (IMAGE batch). Placed if the slot is not already filled by frame_selection_string."}),
                "master_inpaint_mask": ("MASK", {"tooltip": "Optional master inpaint mask. If provided, it defines the entire output mask, overriding masks for selected/depth frames."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("guidance_video_frames", "guidance_frame_masks",)
    FUNCTION = "extract_frames_and_generate_masks"
    CATEGORY = "Steerable-Motion"
    DESCRIPTION = "Extracts/places frames from input/depth video into a new guidance video and generates corresponding masks. frame_selection_string takes precedence over depth frames."

    def _parse_frame_selection_string(self, selection_string, max_frame_index_from_input):
        selected_frame_indices = set()
        selection_parts = selection_string.split(',')
        for part in selection_parts:
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                try:
                    start_str, end_str = part.split(':')
                    start_frame = int(start_str)
                    end_frame = int(end_str)
                    if start_frame < 0 or end_frame < 0:
                        log.warning(f"Frame indices cannot be negative in '{part}'. Skipping.")
                        continue
                    if start_frame > end_frame:
                        log.warning(f"Range start {start_frame} is greater than end {end_frame} in '{part}'. Swapping.")
                        start_frame, end_frame = end_frame, start_frame
                    
                    for frame_idx in range(start_frame, end_frame + 1): # Inclusive range
                        if 0 <= frame_idx <= max_frame_index_from_input:
                            selected_frame_indices.add(frame_idx)
                        else:
                            log.warning(f"Frame index {frame_idx} from range '{part}' is out of bounds for input video (0-{max_frame_index_from_input}). Skipping this specific index.")
                except ValueError:
                    log.error(f"Invalid range format '{part}'. Skipping.")
            else:
                try:
                    frame_idx = int(part)
                    if frame_idx < 0:
                        log.warning(f"Frame index {frame_idx} cannot be negative. Skipping.")
                        continue
                    if 0 <= frame_idx <= max_frame_index_from_input:
                        selected_frame_indices.add(frame_idx)
                    else:
                        log.warning(f"Frame index {frame_idx} is out of bounds for input video (0-{max_frame_index_from_input}). Skipping.")
                except ValueError:
                    log.error(f"Invalid frame index '{part}'. Skipping.")
        return sorted(list(selected_frame_indices))

    def extract_frames_and_generate_masks(self, input_video_frames, total_output_frames, frame_selection_string, empty_frame_fill_level, depth_video_frames=None, master_inpaint_mask=None):
        # Convert string parameter to integer
        total_output_frames = int(total_output_frames)
        if (total_output_frames - 1) % 4 != 0:
            raise ValueError("total_output_frames must satisfy (frames - 1) divisible by 4")
        
        if input_video_frames is None or input_video_frames.shape[0] == 0:
            log.error("Input video_frames is empty. Cannot proceed.")
            dummy_height, dummy_width, dummy_channels = 64, 64, 3
            return (torch.zeros((total_output_frames, dummy_height, dummy_width, dummy_channels), dtype=torch.float32),
                    torch.ones((total_output_frames, dummy_height, dummy_width), dtype=torch.float32))

        device = input_video_frames.device
        dtype = input_video_frames.dtype

        batch_size_input, frame_height, frame_width, num_channels = input_video_frames.shape
        max_input_frame_index = batch_size_input - 1

        # Initialize guidance video with empty_frame_fill_level
        guidance_video_output = torch.ones((total_output_frames, frame_height, frame_width, num_channels), device=device, dtype=dtype) * empty_frame_fill_level
        # Initialize base masks: 1 for unknown/inpaint, 0 for known
        base_frame_masks = torch.ones((total_output_frames, frame_height, frame_width), device=device, dtype=dtype)

        # 1. Process frame_selection_string (highest priority)
        selected_input_frame_indices = self._parse_frame_selection_string(frame_selection_string, max_input_frame_index)
        log.info(f"Frames selected by 'frame_selection_string': {selected_input_frame_indices}")
        for input_frame_index in selected_input_frame_indices:
            # The selected_input_frame_indices are indices from the *input_video*.
            # We place them at the *same index* in the output guidance_video if that index is valid.
            target_output_frame_index = input_frame_index
            if target_output_frame_index < total_output_frames:
                guidance_video_output[target_output_frame_index] = input_video_frames[input_frame_index].clone()
                base_frame_masks[target_output_frame_index] = 0.0 # This frame is now known and prioritized
                log.debug(f"Placed frame {input_frame_index} from input_video_frames into guidance_video at index {target_output_frame_index}.")
            else:
                log.warning(f"Selected frame index {input_frame_index} from input video maps to target index {target_output_frame_index}, which is >= total_output_frames ({total_output_frames}). It won't be placed.")

        # 2. Process depth_video_frames (second priority)
        if depth_video_frames is not None and depth_video_frames.shape[0] > 0:
            log.info(f"Processing {depth_video_frames.shape[0]} depth_video_frames.")
            processed_depth_frames = depth_video_frames.clone().to(device=device, dtype=dtype)

            # Resize depth_video_frames if their dimensions don't match input_video_frames
            if processed_depth_frames.shape[1:] != (frame_height, frame_width, num_channels):
                log.info(f"Resizing depth_video_frames from {processed_depth_frames.shape[1:]} to {(frame_height, frame_width, num_channels)} to match input_video_frames.")
                resized_depth_frame_list = []
                for frame_idx in range(processed_depth_frames.shape[0]):
                    # common_upscale expects (B, C, H, W) or (B, H, W)
                    # IMAGE is (B,H,W,C), so permute, upscale, permute back
                    frame_to_resize = processed_depth_frames[frame_idx:frame_idx+1].permute(0, 3, 1, 2) # (1, C, H_depth, W_depth)
                    resized_frame = common_upscale(frame_to_resize, frame_width, frame_height, "lanczos", "disabled") # (1, C, H, W)
                    resized_depth_frame_list.append(resized_frame.permute(0, 2, 3, 1)) # (1, H, W, C)
                processed_depth_frames = torch.cat(resized_depth_frame_list, dim=0)
            
            num_depth_frames_to_place = min(processed_depth_frames.shape[0], total_output_frames)
            for frame_idx in range(num_depth_frames_to_place):
                # Check if this slot in guidance_video is still an "empty" placeholder
                # (i.e., its mask is still 1.0, meaning not filled by frame_selection_string)
                if base_frame_masks[frame_idx].mean() > 0.99: # Check if it's still (mostly) 1.0
                    guidance_video_output[frame_idx] = processed_depth_frames[frame_idx].clone()
                    # Keep mask as 1.0 for depth frames (inpaint area) - don't set to 0.0
                    log.debug(f"Placed frame {frame_idx} from depth_video_frames into guidance_video at index {frame_idx} (keeping as inpaint area).")
                else:
                    log.debug(f"Skipping depth_frame {frame_idx} as guidance_video index {frame_idx} was already filled by frame_selection_string.")
        else:
            log.info("No depth_video_frames provided or depth_video_frames is empty.")

        # 3. Handle optional master_inpaint_mask (this will override base_frame_masks if provided)
        final_frame_masks = base_frame_masks
        if master_inpaint_mask is not None:
            log.info("Processing provided master_inpaint_mask. This will override masks derived from frame/depth selection.")
            processed_master_mask = master_inpaint_mask.clone().to(device=device, dtype=dtype)

            if processed_master_mask.shape[1:] != (frame_height, frame_width):
                log.info(f"Resizing master_inpaint_mask from {processed_master_mask.shape[1:]} to {(frame_height, frame_width)}.")
                processed_master_mask = common_upscale(
                    processed_master_mask.unsqueeze(1),
                    frame_width, frame_height, "nearest-exact", "disabled"
                ).squeeze(1)

            if processed_master_mask.shape[0] != total_output_frames:
                log.info(f"Adjusting master_inpaint_mask frame count from {processed_master_mask.shape[0]} to {total_output_frames}.")
                if processed_master_mask.shape[0] == 0:
                    log.error("Received an empty master_inpaint_mask after processing. Using base masks.")
                elif processed_master_mask.shape[0] < total_output_frames:
                    num_mask_repeats = (total_output_frames + processed_master_mask.shape[0] - 1) // processed_master_mask.shape[0]
                    processed_master_mask = processed_master_mask.repeat(num_mask_repeats, 1, 1)[:total_output_frames]
                else:
                    processed_master_mask = processed_master_mask[:total_output_frames]
            
            final_frame_masks = processed_master_mask

        return (guidance_video_output.cpu().float(), final_frame_masks.cpu().float())

class VideoContinuationGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video_frames": ("IMAGE", {"tooltip": "Input video frames to create continuation from."}),
                "total_output_frames": ("INT", {"default": 81, "min": 1, "max": 81, "step": 4, "tooltip": "Total number of frames for the output continuation video. Must satisfy: (frames - 1) divisible by 4."}),
                "overlap_frames": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1, "tooltip": "Number of frames from the end of input video to use as overlap at the start."}),
                "empty_frame_fill_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Grayscale level (0.0 black, 1.0 white) for empty continuation frames."}),
            },
            "optional": {
                "end_frame": ("IMAGE", {"tooltip": "Optional single frame to place at the end of the continuation video."}),
                "control_images": ("IMAGE", {"tooltip": "Optional control images to fill the empty frames."}),
                "inpaint_mask": ("MASK", {"tooltip": "Optional inpaint mask to use for the empty frames, overriding the default mask."}),
                "how_to_use_control_images": (["start_sequence_at_beginning_and_prioritise_input_frames", "start_sequence_after_overlap_frames_and_prioritise_input_frames"], {"default": "start_sequence_at_beginning_and_prioritise_input_frames", "tooltip": "If start_sequence_at_beginning_and_prioritise_input_frames is selected, control images align with frame 0 but input overlap frames take priority, so control images become visible after the overlap period. If start_sequence_after_overlap_frames_and_prioritise_input_frames is selected, control images start being placed after the overlap frames from the input video."}),
                "how_to_use_inpaint_masks": (["start_sequence_at_beginning_and_prioritise_input_frames", "start_sequence_after_overlap_frames_and_prioritise_input_frames"], {"default": "start_sequence_at_beginning_and_prioritise_input_frames", "tooltip": "If start_sequence_at_beginning_and_prioritise_input_frames is selected, inpaint masks align with frame 0 but preserve input overlap frames as known. If start_sequence_after_overlap_frames_and_prioritise_input_frames is selected, inpaint masks only affect frames after the overlap period."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("continuation_video_frames", "continuation_frame_masks",)
    FUNCTION = "generate_continuation_video"
    CATEGORY = "Steerable-Motion"
    DESCRIPTION = "Creates a continuation video by placing overlap frames from the end of input video at the start, with optional end frame."

    def generate_continuation_video(self, input_video_frames, total_output_frames, overlap_frames, empty_frame_fill_level, end_frame=None, control_images=None, inpaint_mask=None, how_to_use_control_images="start_sequence_at_beginning_and_prioritise_input_frames", how_to_use_inpaint_masks="start_sequence_at_beginning_and_prioritise_input_frames"):
        # 1. Validation and Setup
        total_output_frames = int(total_output_frames)
        if (total_output_frames - 1) % 4 != 0:
            raise ValueError("total_output_frames must satisfy (frames - 1) divisible by 4")
        
        if input_video_frames is None or input_video_frames.shape[0] == 0:
            log.error("Input video_frames is empty. Cannot proceed.")
            dummy_height, dummy_width, dummy_channels = 64, 64, 3
            return (torch.zeros((total_output_frames, dummy_height, dummy_width, dummy_channels), dtype=torch.float32),
                    torch.ones((total_output_frames, dummy_height, dummy_width), dtype=torch.float32))

        device = input_video_frames.device
        dtype = input_video_frames.dtype
        batch_size_input, frame_height, frame_width, num_channels = input_video_frames.shape

        # 2. Prepare Start Frames (from overlap)
        actual_overlap_frames = min(overlap_frames, batch_size_input, total_output_frames)
        if actual_overlap_frames < overlap_frames:
            log.warning(f"Requested {overlap_frames} overlap frames but input video only has {batch_size_input} frames or total output is smaller. Using {actual_overlap_frames} instead.")
        
        overlap_start_idx = batch_size_input - actual_overlap_frames
        start_frames_part = input_video_frames[overlap_start_idx : overlap_start_idx + actual_overlap_frames].clone()

        # 3. Prepare End Frame
        end_frame_part = torch.empty((0, frame_height, frame_width, num_channels), device=device, dtype=dtype)
        num_end_frames = 0
        if end_frame is not None and end_frame.shape[0] > 0 and total_output_frames > actual_overlap_frames:
            num_end_frames = 1
            end_frame_processed = end_frame[0].clone().to(device=device, dtype=dtype)
            
            if end_frame_processed.shape != (frame_height, frame_width, num_channels):
                log.info(f"Resizing end_frame from {end_frame_processed.shape} to {(frame_height, frame_width, num_channels)}.")
                frame_to_resize = end_frame_processed.unsqueeze(0).permute(0, 3, 1, 2)
                resized_frame = common_upscale(frame_to_resize, frame_width, frame_height, "lanczos", "disabled")
                end_frame_processed = resized_frame.permute(0, 2, 3, 1).squeeze(0)
            
            end_frame_part = end_frame_processed.unsqueeze(0)

        # 4. Prepare Middle Frames
        num_middle_frames = total_output_frames - actual_overlap_frames - num_end_frames
        middle_frames_part = torch.empty((0, frame_height, frame_width, num_channels), device=device, dtype=dtype)

        if num_middle_frames > 0:
            if control_images is not None:
                log.info(f"Using 'control_images' to fill the {num_middle_frames} middle frames with '{how_to_use_control_images}' mode.")
                control_images_resized = common_upscale(control_images.movedim(-1, 1), frame_width, frame_height, "lanczos", "disabled").movedim(1, -1)
                
                if how_to_use_control_images == "start_sequence_at_beginning_and_prioritise_input_frames":
                    # Skip the first overlap_frames control images to avoid duplication
                    duplicate_count = min(actual_overlap_frames, control_images_resized.shape[0])
                    available_after_dup = control_images_resized.shape[0] - duplicate_count
                    if available_after_dup < num_middle_frames:
                        log.info(f"After skipping {duplicate_count} control frames, only {available_after_dup} remain; padding {num_middle_frames - available_after_dup} frames with 'empty_frame_fill_level'.")
                        selected_control = control_images_resized[duplicate_count:]
                        padding_needed = num_middle_frames - selected_control.shape[0]
                        padding = torch.ones((padding_needed, frame_height, frame_width, num_channels), device=device, dtype=dtype) * empty_frame_fill_level
                        middle_frames_part = torch.cat([selected_control, padding], dim=0)
                    else:
                        middle_frames_part = control_images_resized[duplicate_count:duplicate_count + num_middle_frames].clone()
                else:  # "start_sequence_after_overlap_frames_and_prioritise_input_frames"
                    # Use control frames from the beginning of the sequence (C0, C1, C2...)
                    if control_images_resized.shape[0] < num_middle_frames:
                        log.warning(f"Provided 'control_images' have {control_images_resized.shape[0]} frames, less than needed ({num_middle_frames}). Padding with 'empty_frame_fill_level'.")
                        padding_needed = num_middle_frames - control_images_resized.shape[0]
                        padding = torch.ones((padding_needed, frame_height, frame_width, num_channels), device=device, dtype=dtype) * empty_frame_fill_level
                        middle_frames_part = torch.cat([control_images_resized, padding], dim=0)
                    else:
                        middle_frames_part = control_images_resized[:num_middle_frames].clone()
            else:
                log.info(f"No 'control_images', filling {num_middle_frames} middle frames with level {empty_frame_fill_level}.")
                middle_frames_part = torch.ones((num_middle_frames, frame_height, frame_width, num_channels), device=device, dtype=dtype) * empty_frame_fill_level
        
        # 5. Assemble Final Video
        continuation_video_output = torch.cat([start_frames_part, middle_frames_part, end_frame_part], dim=0)
        
        # 6. Create Mask
        continuation_frame_masks = torch.ones((total_output_frames, frame_height, frame_width), device=device, dtype=dtype)
        
        # Apply mask logic based on how_to_use_inpaint_masks parameter
        if how_to_use_inpaint_masks == "start_sequence_at_beginning_and_prioritise_input_frames":
            # Set known frames (overlap and end) to 0.0, but also set middle section based on control frame logic
            if actual_overlap_frames > 0:
                continuation_frame_masks[0:actual_overlap_frames] = 0.0
            if num_end_frames > 0:
                continuation_frame_masks[-num_end_frames:] = 0.0
            
            # For middle section, follow the same logic as control frames
            if control_images is not None and num_middle_frames > 0:
                duplicate_count = min(actual_overlap_frames, control_images.shape[0])
                available_after_dup = control_images.shape[0] - duplicate_count
                if available_after_dup >= num_middle_frames:
                    # If we have enough control frames after skipping, set those middle frames as known (0.0)
                    middle_start = actual_overlap_frames
                    middle_end = middle_start + num_middle_frames
                    continuation_frame_masks[middle_start:middle_end] = 0.0
        else:  # "start_sequence_after_overlap_frames_and_prioritise_input_frames"
            # Set known frames (overlap and end) to 0.0, rest stay as 1.0 (inpaint)
            if actual_overlap_frames > 0:
                continuation_frame_masks[0:actual_overlap_frames] = 0.0
            if num_end_frames > 0:
                continuation_frame_masks[-num_end_frames:] = 0.0
        
        # 7. Handle optional inpaint_mask with how_to_use_inpaint_masks logic
        if inpaint_mask is not None:
            log.info(f"Processing provided 'inpaint_mask' with '{how_to_use_inpaint_masks}' timing.")
            processed_mask = common_upscale(inpaint_mask.unsqueeze(1), frame_width, frame_height, "nearest-exact", "disabled").squeeze(1).to(device)
            
            if processed_mask.shape[0] != total_output_frames:
                log.info(f"Adjusting inpaint_mask frame count from {processed_mask.shape[0]} to {total_output_frames}.")
                if processed_mask.shape[0] < total_output_frames:
                    num_repeats = (total_output_frames + processed_mask.shape[0] - 1) // processed_mask.shape[0]
                    processed_mask = processed_mask.repeat(num_repeats, 1, 1)[:total_output_frames]
                else:
                    processed_mask = processed_mask[:total_output_frames]
            
            # Apply how_to_use_inpaint_masks logic to the provided mask
            if how_to_use_inpaint_masks == "start_sequence_at_beginning_and_prioritise_input_frames":
                # Use the provided mask as-is, but preserve known frames (overlap and end)
                if actual_overlap_frames > 0:
                    processed_mask[0:actual_overlap_frames] = 0.0  # Keep overlap frames as known
                if num_end_frames > 0:
                    processed_mask[-num_end_frames:] = 0.0  # Keep end frame as known
            else:  # "start_sequence_after_overlap_frames_and_prioritise_input_frames"
                # Only apply the provided mask after overlap frames
                if actual_overlap_frames > 0:
                    processed_mask[0:actual_overlap_frames] = 0.0  # Keep overlap frames as known
                    # The provided mask affects frames starting after overlap
                if num_end_frames > 0:
                    processed_mask[-num_end_frames:] = 0.0  # Keep end frame as known
            
            continuation_frame_masks = processed_mask.to(dtype=dtype)

        log.info(f"Generated continuation video. Start: {actual_overlap_frames} frames, Middle: {num_middle_frames} frames, End: {num_end_frames} frames.")
        
        return (continuation_video_output.cpu().float(), continuation_frame_masks.cpu().float())

class WanInputFrameNumber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_number": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Frame number where (frames - 1) is divisible by 4."}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_number",)
    FUNCTION = "get_frame_number"
    CATEGORY = "Steerable-Motion"
    DESCRIPTION = "Outputs a frame number that satisfies the WAN constraint: (frames - 1) divisible by 4."

    def get_frame_number(self, frame_number):
        frame_number = int(frame_number)
        if (frame_number - 1) % 4 != 0:
            raise ValueError("frame_number must satisfy (frame_number - 1) divisible by 4")
        return (frame_number,)

class WanVideoBlenderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "overlap_frames": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
                "video_1": ("IMAGE",),
                "video_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_video_frames",)
    FUNCTION = "blend_videos"
    CATEGORY = "Steerable-Motion"
    DESCRIPTION = "Blends two input videos with a cross-fade. The resolution of the second clip is resized to match the first."

    def _resize_video(self, video, target_height, target_width):
        """Resize a batch of frames (B,H,W,C) to (target_height,target_width) using Lanczos."""
        if video.shape[1] == target_height and video.shape[2] == target_width:
            return video
        # (B, H, W, C) -> (B, C, H, W)
        video_permuted = video.permute(0, 3, 1, 2)
        resized = common_upscale(video_permuted, target_width, target_height, "lanczos", "disabled")  # (B, C, H, W)
        return resized.permute(0, 2, 3, 1)

    def _cross_fade(self, tail, head, overlap_frames):
        """Blend two tensors of shape (overlap_frames,H,W,C) using linear alpha."""
        device, dtype = tail.device, tail.dtype
        alphas = torch.linspace(0, 1, overlap_frames, device=device, dtype=dtype).view(-1, 1, 1, 1)
        blended = tail * (1 - alphas) + head * alphas
        return blended

    def blend_videos(self, overlap_frames, video_1, video_2):
        if video_1 is None or video_2 is None:
            raise ValueError("Both video_1 and video_2 are required.")

        # Reference dimensions and properties from first video
        ref_h, ref_w = video_1.shape[1:3]
        
        # Ensure second video matches size
        video_2_resized = self._resize_video(video_2, ref_h, ref_w)

        if video_1.shape[0] < overlap_frames or video_2_resized.shape[0] < overlap_frames:
            raise ValueError(f"One of the videos is shorter than overlap_frames={overlap_frames}.")

        # Extract segments for blending
        tail = video_1[-overlap_frames:]
        head = video_2_resized[:overlap_frames]
        blended = self._cross_fade(tail, head, overlap_frames)

        # Assemble new timeline
        final_video = torch.cat([
            video_1[:-overlap_frames],
            blended,
            video_2_resized[overlap_frames:]
        ], dim=0)

        return (final_video.cpu().float(),)

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "BatchCreativeInterpolation": BatchCreativeInterpolationNode,
    "IpaConfiguration": IpaConfigurationNode,
    "RemoveAndInterpolateFrames": RemoveAndInterpolateFramesNode,
    "VideoFrameExtractorAndMaskGenerator": VideoFrameExtractorAndMaskGenerator,
    "VideoContinuationGenerator": VideoContinuationGenerator,
    "WanInputFrameNumber": WanInputFrameNumber,
    "WanVideoBlender": WanVideoBlenderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {    
    "BatchCreativeInterpolation": "Batch Creative Interpolation 🎞️🅢🅜",
    "IpaConfiguration": "IP-Adapter Configuration 🎞️🅢🅜",
    "RemoveAndInterpolateFrames": "Remove and Interpolate Frames 🎞️🅢🅜",
    "VideoFrameExtractorAndMaskGenerator": "Video Frame Extractor & Mask Generator 🎞️🅢🅜",
    "VideoContinuationGenerator": "Video Continuation Generator 🎞️🅢🅜",
    "WanInputFrameNumber": "WAN Input Frame Number 🎞️🅢🅜",
    "WanVideoBlender": "WAN Video Blender 🎞️🅢🅜",
}
