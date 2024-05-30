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
from .imports.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterBatchImport, IPAdapterTiledBatchImport, IPAdapterTiledImport, PrepImageForClipVisionImport, IPAdapterAdvancedImport, IPAdapterNoiseImport
from .imports.ComfyUI_Frame_Interpolation.vfi_models.film import FILM_VFIImport
import matplotlib
import gc

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

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "BatchCreativeInterpolation": BatchCreativeInterpolationNode,
    "IpaConfiguration": IpaConfigurationNode,
    "RemoveAndInterpolateFrames": RemoveAndInterpolateFramesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {    
    "BatchCreativeInterpolation": "Batch Creative Interpolation 🎞️🅢🅜",
    "IpaConfiguration": "IPA Configuration  🎞️🅢🅜",
    "RemoveAndInterpolateFrames": "Remove and Interpolate Frames 🎞️🅢🅜",
}