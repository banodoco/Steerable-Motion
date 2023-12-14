from typing import Union
import numpy as np
from collections.abc import Iterable

from .control import LatentKeyframeImport, LatentKeyframeGroupImport
from .control import StrengthInterpolationImport as SI
from .logger import logger


class LatentKeyframeNodeImport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                      batch_index: int,
                      strength: float,
                      prev_latent_kf: LatentKeyframeGroupImport=None,
                      prev_latent_keyframe: LatentKeyframeGroupImport=None, # old name
                      ):
        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroupImport()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        keyframe = LatentKeyframeImport(batch_index, strength)
        prev_latent_keyframe.add(keyframe)
        return (prev_latent_keyframe,)


class LatentKeyframeGroupNodeImport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index_strengths": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
                "latent_optional": ("LATENT", ),
                "print_keyframes": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframes"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def validate_index(self, index: int, latent_count: int = 0, is_range: bool = False, allow_negative = False) -> int:
        # if part of range, do nothing
        if is_range:
            return index
        # otherwise, validate index
        # validate not out of range - only when latent_count is passed in
        if latent_count > 0 and index > latent_count-1:
            raise IndexError(f"Index '{index}' out of range for the total {latent_count} latents.")
        # if negative, validate not out of range
        if index < 0:
            if not allow_negative:
                raise IndexError(f"Negative indeces not allowed, but was {index}.")
            conv_index = latent_count+index
            if conv_index < 0:
                raise IndexError(f"Index '{index}', converted to '{conv_index}' out of range for the total {latent_count} latents.")
            index = conv_index
        return index

    def convert_to_index_int(self, raw_index: str, latent_count: int = 0, is_range: bool = False, allow_negative = False) -> int:
        try:
            return self.validate_index(int(raw_index), latent_count=latent_count, is_range=is_range, allow_negative=allow_negative)
        except ValueError as e:
            raise ValueError(f"index '{raw_index}' must be an integer.", e)

    def convert_to_latent_keyframes(self, latent_indeces: str, latent_count: int) -> set[LatentKeyframeImport]:
        if not latent_indeces:
            return set()
        int_latent_indeces = [i for i in range(0, latent_count)]
        allow_negative = latent_count > 0
        chosen_indeces = set()
        # parse string - allow positive ints, negative ints, and ranges separated by ':'
        groups = latent_indeces.split(",")
        groups = [g.strip() for g in groups]
        for g in groups:
            # parse strengths - default to 1.0 if no strength given
            strength = 1.0
            if '=' in g:
                g, strength_str = g.split("=", 1)
                g = g.strip()
                try:
                    strength = float(strength_str.strip())
                except ValueError as e:
                    raise ValueError(f"strength '{strength_str}' must be a float.", e)
                if strength < 0:
                    raise ValueError(f"Strength '{strength}' cannot be negative.")
            # parse range of indeces (e.g. 2:16)
            if ':' in g:
                index_range = g.split(":", 1)
                index_range = [r.strip() for r in index_range]
                start_index = self.convert_to_index_int(index_range[0], latent_count=latent_count, is_range=True, allow_negative=allow_negative)
                end_index = self.convert_to_index_int(index_range[1], latent_count=latent_count, is_range=True, allow_negative=allow_negative)
                # if latents were passed in, base indeces on known latent count
                if len(int_latent_indeces) > 0:
                    for i in int_latent_indeces[start_index:end_index]:
                        chosen_indeces.add(LatentKeyframeImport(i, strength))
                # otherwise, assume indeces are valid
                else:
                    for i in range(start_index, end_index):
                        chosen_indeces.add(LatentKeyframeImport(i, strength))
            # parse individual indeces
            else:
                chosen_indeces.add(LatentKeyframeImport(self.convert_to_index_int(g, latent_count=latent_count, allow_negative=allow_negative), strength))
        return chosen_indeces

    def load_keyframes(self,
                       index_strengths: str,
                       prev_latent_kf: LatentKeyframeGroupImport=None,
                       prev_latent_keyframe: LatentKeyframeGroupImport=None, # old name
                       latent_image_opt=None,
                       print_keyframes=False):
        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroupImport()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroupImport()

        latent_count = -1
        if latent_image_opt:
            latent_count = latent_image_opt['samples'].size()[0]
        latent_keyframes = self.convert_to_latent_keyframes(index_strengths, latent_count=latent_count)

        for latent_keyframe in latent_keyframes:
            curr_latent_keyframe.add(latent_keyframe)
        
        if print_keyframes:
            for keyframe in curr_latent_keyframe.keyframes:
                logger.info(f"keyframe {keyframe.batch_index}:{keyframe.strength}")

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)

        
class LatentKeyframeInterpolationNodeImport:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index_from": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "batch_index_to_excl": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "strength_from": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}, ),
                "strength_to": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}, ),
                "interpolation": (["linear", "ease-in", "ease-out", "ease-in-out"], ),
                "revert_direction_at_midpoint": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_latent_keyframe": ("LATENT_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                        batch_index_from: int,
                        strength_from: float,
                        batch_index_to_excl: int,
                        strength_to: float,
                        interpolation: str,
                        revert_direction_at_midpoint: bool=False,
                        last_key_frame_position: int=0,
                        i=0,
                        number_of_items=0,
                        buffer=0,
                        prev_latent_keyframe: LatentKeyframeGroupImport=None):



        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroupImport()
        else:             
            prev_latent_keyframe = prev_latent_keyframe.clone()
        
        curr_latent_keyframe = LatentKeyframeGroupImport()

        weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, strength_from, strength_to, interpolation, revert_direction_at_midpoint, last_key_frame_position,i,number_of_items, buffer)
        
        for i, frame_number in enumerate(frame_numbers):
            keyframe = LatentKeyframeImport(frame_number, float(weights[i]))            
            curr_latent_keyframe.add(keyframe)

        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)


        return (weights, frame_numbers, curr_latent_keyframe,)

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

    # If it's a middle keyframe, mirror the weights
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

class LatentKeyframeBatchedGroupNodeImport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_strengths": ("FLOAT", {"default": -1, "min": -1, "step": 0.001, "forceInput": True}),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
                "print_keyframes": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self, float_strengths: Union[float, list[float]],
                      prev_latent_kf: LatentKeyframeGroupImport=None,
                      prev_latent_keyframe: LatentKeyframeGroupImport=None, # old name
                      print_keyframes=False):
        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroupImport()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroupImport()

        # if received a normal float input, do nothing
        if type(float_strengths) in (float, int):
            logger.info("No batched float_strengths passed into Latent Keyframe Batch Group node; will not create any new keyframes.")
        # if iterable, attempt to create LatentKeyframes with chosen strengths
        elif isinstance(float_strengths, Iterable):
            for idx, strength in enumerate(float_strengths):
                keyframe = LatentKeyframeImport(idx, strength)
                curr_latent_keyframe.add(keyframe)
        else:
            raise ValueError(f"Expected strengths to be an iterable input, but was {type(float_strengths).__repr__}.")    

        if print_keyframes:
            for keyframe in curr_latent_keyframe.keyframes:
                logger.info(f"keyframe {keyframe.batch_index}:{keyframe.strength}")

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)
