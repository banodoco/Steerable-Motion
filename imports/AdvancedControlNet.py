from typing import Union

from collections.abc import Iterable
import folder_paths
import torch
import numpy as np
from torch import Tensor
from comfy.controlnet import  ControlNet, T2IAdapter,broadcast_image_to
import comfy.utils
import comfy.controlnet as comfy_cn

ControlNetWeightsTypeImport = list[float]
T2IAdapterWeightsTypeImport = list[float]

    
class LatentKeyframeImport:
    def __init__(self, batch_index: int, strength: float) -> None:
        self.batch_index = batch_index
        self.strength = strength

class LatentKeyframeGroupImport:
    def __init__(self) -> None:
        self.keyframes: list[LatentKeyframeImport] = []

    def add(self, keyframe: LatentKeyframeImport) -> None:
        added = False
        # replace existing keyframe if same batch_index
        for i in range(len(self.keyframes)):
            if self.keyframes[i].batch_index == keyframe.batch_index:
                self.keyframes[i] = keyframe
                added = True
                break
        if not added:
            self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.batch_index)
    
    def get_index(self, index: int) -> Union[LatentKeyframeImport, None]:
        try:
            return self.keyframes[index]
        except IndexError:
            return None
    
    def __getitem__(self, index) -> LatentKeyframeImport:
        return self.keyframes[index]
    
    def is_empty(self) -> bool:
        return len(self.keyframes) == 0

class TimestepKeyframeImport:
    def __init__(self,
                 start_percent: float = 0.0,
                 control_net_weights: ControlNetWeightsTypeImport = None,
                 t2i_adapter_weights: T2IAdapterWeightsTypeImport = None,
                 latent_keyframes: LatentKeyframeGroupImport = None,
                 default_latent_strength: float = 0.0) -> None:
        self.start_percent = start_percent
        self.control_net_weights = control_net_weights
        self.t2i_adapter_weights = t2i_adapter_weights
        self.latent_keyframes = latent_keyframes
        self.default_latent_strength = default_latent_strength
    
    
    @classmethod
    def default(cls) -> 'TimestepKeyframeImport':
        return cls(0.0)
    
class TimestepKeyframeGroupImport:
    def __init__(self) -> None:
        self.keyframes: list[TimestepKeyframeImport] = []
        self.keyframes.append(TimestepKeyframeImport.default())

    def add(self, keyframe: TimestepKeyframeImport) -> None:
        added = False
        # replace existing keyframe if same start_percent
        for i in range(len(self.keyframes)):
            if self.keyframes[i].start_percent == keyframe.start_percent:
                self.keyframes[i] = keyframe
                added = True
                break
        if not added:
            self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.start_percent)

    def get_index(self, index: int) -> Union[TimestepKeyframeImport, None]:
        try:
            return self.keyframes[index]
        except IndexError:
            return None
    
    def __getitem__(self, index) -> TimestepKeyframeImport:
        return self.keyframes[index]
    
    def is_empty(self) -> bool:
        return len(self.keyframes) == 0
    
    @classmethod
    def default(cls, keyframe: TimestepKeyframeImport) -> 'TimestepKeyframeGroupImport':
        group = cls()
        group.keyframes[0] = keyframe
        return group




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
    
class LatentKeyframeGroupNodeImport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index_strengths": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prev_latent_keyframe": ("LATENT_KEYFRAME", ),
                "latent_optional": ("LATENT", ),
            }
        }
    
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
        all_indeces = [i for i in range(0, latent_count)]
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
                for i in all_indeces[start_index:end_index]:
                    chosen_indeces.add(LatentKeyframeImport(i, strength))
            # parse individual indeces
            else:
                chosen_indeces.add(LatentKeyframeImport(self.convert_to_index_int(g, latent_count=latent_count, allow_negative=allow_negative), strength))
        return chosen_indeces

    def load_keyframes(self,
                       index_strengths: str,
                       prev_latent_keyframe: LatentKeyframeGroupImport=None,
                       latent_image_opt=None):
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroupImport()
        curr_latent_keyframe = LatentKeyframeGroupImport()

        latent_count = -1
        if latent_image_opt:
            latent_count = latent_image_opt['samples'].size()[0]
        latent_keyframes = self.convert_to_latent_keyframes(index_strengths, latent_count=latent_count)

        for latent_keyframe in latent_keyframes:
            
            curr_latent_keyframe.add(latent_keyframe)
        
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
        
        curr_latent_keyframe = LatentKeyframeGroupImport()

        weights, frame_numbers = calculate_weights(batch_index_from, batch_index_to_excl, strength_from, strength_to, interpolation, revert_direction_at_midpoint, last_key_frame_position,i,number_of_items, buffer)
        
        for i, frame_number in enumerate(frame_numbers):
            keyframe = LatentKeyframeImport(frame_number, float(weights[i]))            
            curr_latent_keyframe.add(keyframe)

        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)


        return (weights, frame_numbers, curr_latent_keyframe,)

class ControlNetAdvancedImport(ControlNet):
    def __init__(self, control_model, timestep_keyframes: TimestepKeyframeGroupImport, global_average_pooling=False, device=None):
        super().__init__(control_model=control_model, global_average_pooling=global_average_pooling, device=device)
        # initialize timestep_keyframes
        self.timestep_keyframes = timestep_keyframes if timestep_keyframes else TimestepKeyframeGroupImport()
        self.current_timestep_keyframe = self.timestep_keyframes.keyframes[0]
        # initialize weights
        self.weights = self.timestep_keyframes.keyframes[0].control_net_weights if self.timestep_keyframes.keyframes[0].control_net_weights else [1.0]*13
        # mask for which parts of controlnet output to keep
        self.mask_cond_hint_original = None
        self.mask_cond_hint = None
        # actual index values
        self.sub_idxs = None
        self.full_latent_length = 0
        self.context_length = 0
        # override control_merge
        self.control_merge = control_merge_inject.__get__(self, type(self))

    def set_cond_hint_mask(self, mask_hint):
        self.mask_cond_hint_original = mask_hint
        return self

    def get_control(self, x_noisy, t, cond, batched_number):
        # need to reference t and batched_number later
        self.t = t
        self.batched_number = batched_number
        # TODO: choose TimestepKeyframe based on t
        # perform special version of get_control that supports sliding context and masks
        return self.sliding_get_control(x_noisy, t, cond, batched_number)

    def sliding_get_control(self, x_noisy: Tensor, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        output_dtype = x_noisy.dtype

        # make cond_hint appropriate dimensions
        # TODO: change this to not require cond_hint upscaling every step when self.sub_idxs are present
        if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            # if self.cond_hint_original length matches real latent count, need to subdivide it
            if self.cond_hint_original.size(0) == self.full_latent_length:
                self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original[self.sub_idxs], x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(self.control_model.dtype).to(self.device)
            else:
                self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(self.control_model.dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        # make mask appropriate dimensions, if present
        if self.mask_cond_hint_original is not None:
            if self.sub_idxs is not None or self.mask_cond_hint is None or x_noisy.shape[2] * 8 != self.mask_cond_hint.shape[1] or x_noisy.shape[3] * 8 != self.mask_cond_hint.shape[2]:
                if self.mask_cond_hint is not None:
                    del self.mask_cond_hint
                self.mask_cond_hint = None
                # TODO: perform upscale on only the sub_idxs masks at a time instead of all to conserve RAM
                # resize mask and match batch count
                self.mask_cond_hint = prepare_mask_batch(self.mask_cond_hint_original, x_noisy.shape, multiplier=8)
                actual_latent_length = x_noisy.shape[0] // batched_number
                self.mask_cond_hint = comfy.utils.repeat_to_batch_size(self.mask_cond_hint, actual_latent_length if self.sub_idxs is None else self.full_latent_length)
                if self.sub_idxs is not None:
                    self.mask_cond_hint = self.mask_cond_hint[self.sub_idxs]
            # make cond_hint_mask length match x_noise
            if x_noisy.shape[0] != self.mask_cond_hint.shape[0]:
                self.mask_cond_hint = broadcast_image_to(self.mask_cond_hint, x_noisy.shape[0], batched_number)
            self.mask_cond_hint = self.mask_cond_hint.to(self.control_model.dtype).to(self.device)

        context = cond['c_crossattn']
        # uses 'y' in new ComfyUI update
        y = cond.get('y', None)
        if y is None: # TODO: remove this in the future since no longer used by newest ComfyUI
            y = cond.get('c_adm', None)
        if y is not None:
            y = y.to(self.control_model.dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(self.control_model.dtype), hint=self.cond_hint, timesteps=timestep.float(), context=context.to(self.control_model.dtype), y=y)
        return self.control_merge(None, control, control_prev, output_dtype)

    def apply_advanced_strengths_and_masks(self, x: Tensor, current_timestep_keyframe: TimestepKeyframeImport, batched_number: int):
        # apply strengths, and get batch indeces to default out
        # AKA latents that should not be influenced by ControlNet
        if current_timestep_keyframe.latent_keyframes is not None:
            latent_count = x.size(0)//batched_number
            indeces_to_default = set(range(latent_count))
            mapped_indeces = None
            # if expecting subdivision, will need to translate between subset and actual idx values
            if self.sub_idxs:
                mapped_indeces = {}
                for i, actual in enumerate(self.sub_idxs):
                    mapped_indeces[actual] = i
            for keyframe in current_timestep_keyframe.latent_keyframes:
                real_index = keyframe.batch_index
                # if negative, count from end
                if real_index < 0:
                    real_index += latent_count if self.sub_idxs is None else self.full_latent_length

                # if not mapping indeces, what you see is what you get
                if mapped_indeces is None:
                    if real_index in indeces_to_default:
                        indeces_to_default.remove(real_index)
                # otherwise, see if batch_index is even included in this set of latents
                else:
                    real_index = mapped_indeces.get(real_index, None)
                    if real_index is None:
                        continue
                    indeces_to_default.remove(real_index)

                # apply strength for each batched cond/uncond
                for b in range(batched_number):
                    x[(latent_count*b)+real_index] = x[(latent_count*b)+real_index] * keyframe.strength

            # default them out by multiplying by default_latent_strength
            for batch_index in indeces_to_default:
                # apply default for each batched cond/uncond
                for b in range(batched_number):
                    x[(latent_count*b)+batch_index] = x[(latent_count*b)+batch_index] * current_timestep_keyframe.default_latent_strength
        # apply masks
        if self.mask_cond_hint is not None:
            # first, resize mask to required dims
            masks = prepare_mask_batch(self.mask_cond_hint, x.shape)
            x[:] = x[:] * masks

    def copy(self):
        c = ControlNetAdvancedImport(self.control_model, self.timestep_keyframes, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def cleanup(self):
        super().cleanup()
        self.sub_idxs = None
        self.full_latent_length = 0
        self.context_length = 0

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

class ScaledSoftControlNetWeightsImport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_multiplier": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    FUNCTION = "load_weights"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights"

    def load_weights(self, base_multiplier, flip_weights):
        weights = [(base_multiplier ** float(12 - i)) for i in range(13)]
        if flip_weights:
            weights.reverse()
        return (weights, TimestepKeyframeGroupImport.default(TimestepKeyframeImport(control_net_weights=weights)))

class LatentKeyframeBatchedGroupNodeImport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strengths": ("FLOAT", {"default": -1, "min": -1, "step": 0.0001}),
            },
            "optional": {
                "prev_latent_keyframe": ("LATENT_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self, strengths: Union[float, list[float]], prev_latent_keyframe: LatentKeyframeGroupImport=None):
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroupImport()
        curr_latent_keyframe = LatentKeyframeGroupImport()

        # if received a normal float input, do nothing
        if type(strengths) in (float, int):
            print("No batched strengths passed into Latent Keyframe Batch Group node; will not create any new keyframes.")
        # if iterable, attempt to create LatentKeyframes with chosen strengths
        elif isinstance(strengths, Iterable):
            for idx, strength in enumerate(strengths):
                keyframe = LatentKeyframeImport(idx, strength)
                curr_latent_keyframe.add(keyframe)                
        else:
            raise ValueError(f"Expected strengths to be an iterable input, but was {type(strengths).__repr__}.")    

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)

class T2IAdapterAdvancedImport(T2IAdapter):
    def __init__(self, t2i_model, timestep_keyframes: TimestepKeyframeGroupImport, channels_in, device=None):
        super().__init__(t2i_model=t2i_model, channels_in=channels_in, device=device)
        self.timestep_keyframes = timestep_keyframes if timestep_keyframes else TimestepKeyframeGroupImport()
        self.current_timestep_keyframe = self.timestep_keyframes.keyframes[0]
        first_weight = self.timestep_keyframes.keyframes[0].t2i_adapter_weights if self.timestep_keyframes.get_index(0) else None
        self.weights = first_weight if first_weight else [1.0]*12
        # mask for which parts of controlnet output to keep
        self.cond_hint_mask = None
        # actual index values
        self.sub_idxs = None
        self.full_latent_length = 0
        self.context_length = 0
        # override control_merge
        self.control_merge = control_merge_inject.__get__(self, type(self))
    
    def get_control(self, x_noisy, t, cond, batched_number):
        # need to reference t and batched_number later
        self.t = t
        self.batched_number = batched_number
        # TODO: choose TimestepKeyframe based on t
        try:
            # if sub indexes present, replace original hint with subsection
            if self.sub_idxs is not None:
                full_cond_hint_original = self.cond_hint_original
                del self.cond_hint
                self.cond_hint = None
                self.cond_hint_original = full_cond_hint_original[self.sub_idxs]
            return super().get_control(x_noisy, t, cond, batched_number)
        finally:
            if self.sub_idxs is not None:
                # replace original cond hint
                self.cond_hint_original = full_cond_hint_original
                del full_cond_hint_original

    def apply_advanced_strengths_and_masks(self, x, current_timestep_keyframe: TimestepKeyframeImport, batched_number: int):
        # For now, do nothing; need to figure out LatentKeyframe control is even possible for T2I Adapters
        # TODO: support masks
        return

    def copy(self):
        c = T2IAdapterAdvancedImport(self.t2i_model, self.timestep_keyframes, self.channels_in)
        self.copy_to(c)
        return c
    
    def cleanup(self):
        super().cleanup()
        self.sub_idxs = None
        self.full_latent_length = 0
        self.context_length = 0


def is_advanced_controlnet(input_object):
    return isinstance(input_object, ControlNetAdvancedImport) or isinstance(input_object, T2IAdapterAdvancedImport)

def control_merge_inject(self, control_input, control_output, control_prev, output_dtype):
    out = {'input':[], 'middle':[], 'output': []}

    if control_input is not None:
        for i in range(len(control_input)):
            key = 'input'
            x = control_input[i]
            if x is not None:
                self.apply_advanced_strengths_and_masks(x, self.current_timestep_keyframe, self.batched_number)

                x *= self.strength * self.weights[i]
                if x.dtype != output_dtype:
                    x = x.to(output_dtype)
            out[key].insert(0, x)

    if control_output is not None:
        for i in range(len(control_output)):
            if i == (len(control_output) - 1):
                key = 'middle'
                index = 0
            else:
                key = 'output'
                index = i
            x = control_output[i]
            if x is not None:
                self.apply_advanced_strengths_and_masks(x, self.current_timestep_keyframe, self.batched_number)

                if self.global_average_pooling:
                    x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                x *= self.strength * self.weights[i]
                if x.dtype != output_dtype:
                    x = x.to(output_dtype)

            out[key].append(x)
    if control_prev is not None:
        for x in ['input', 'middle', 'output']:
            o = out[x]
            for i in range(len(control_prev[x])):
                prev_val = control_prev[x][i]
                if i >= len(o):
                    o.append(prev_val)
                elif prev_val is not None:
                    if o[i] is None:
                        o[i] = prev_val
                    else:
                        o[i] += prev_val
    return out

def prepare_mask_batch(mask: Tensor, shape: Tensor, multiplier: int=1, match_dim1=False):
    mask = mask.clone()
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2]*multiplier, shape[3]*multiplier), mode="bilinear")
    if match_dim1:
        mask = torch.cat([mask] * shape[1], dim=1)
    return mask

def load_controlnet(ckpt_path, timestep_keyframe: TimestepKeyframeGroupImport=None, model=None):
    control = comfy_cn.load_controlnet(ckpt_path, model=model)
    # if exactly ControlNet returned, transform it into ControlNetAdvanced
    if type(control) == ControlNet:
        return ControlNetAdvancedImport(control.control_model, timestep_keyframe, global_average_pooling=control.global_average_pooling)
    # if T2IAdapter returned, transform it into T2IAdapterAdvanced
    elif isinstance(control, T2IAdapter):
        return T2IAdapterAdvancedImport(control.t2i_model, timestep_keyframe, control.channels_in)
    # otherwise, leave it be - probably a ControlLora for SDXL (no support for advanced stuff yet from here)
    # TODO add ControlLoraAdvanced
    return control

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