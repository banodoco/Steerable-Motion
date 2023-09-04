import sys
import os


import torch
import contextlib
import copy
import inspect

from ldm.modules.diffusionmodules.util import timestep_embedding

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from comfy.cldm import cldm

from comfy.model_patcher import ModelPatcher
from comfy.controlnet import ControlBase, ControlNet, T2IAdapter, broadcast_image_to, ControlLora
import comfy.t2i_adapter as t2i_adapter
import comfy.utils as utils
import comfy.model_management as model_management
import comfy.model_detection as model_detection

ControlNetWeightsType = list[float]
T2IAdapterWeightsType = list[float]


class LatentKeyframe:
    def __init__(self, batch_index: int, strength: float) -> None:
        self.batch_index = batch_index
        self.strength = strength


# always maintain sorted state (by batch_index of LatentKeyframe)
class LatentKeyframeGroup:
    def __init__(self) -> None:
        self.keyframes: list[LatentKeyframe] = []

    def add(self, keyframe: LatentKeyframe) -> None:
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
    
    def get_index(self, index: int) -> LatentKeyframe | None:
        try:
            return self.keyframes[index]
        except IndexError:
            return None
    
    def __getitem__(self, index) -> LatentKeyframe:
        return self.keyframes[index]
    
    def is_empty(self) -> bool:
        return len(self.keyframes) == 0


class TimestepKeyframe:
    def __init__(self,
                 start_percent: float = 0.0,
                 control_net_weights: ControlNetWeightsType = None,
                 t2i_adapter_weights: T2IAdapterWeightsType = None,
                 latent_keyframes: LatentKeyframeGroup = None) -> None:
        self.start_percent = start_percent
        self.control_net_weights = control_net_weights
        self.t2i_adapter_weights = t2i_adapter_weights
        self.latent_keyframes = latent_keyframes
    
    
    @classmethod
    def default(cls) -> 'TimestepKeyframe':
        return cls(0.0)


# always maintain sorted state (by start_percent of TimestepKeyFrame)
class TimestepKeyframeGroup:
    def __init__(self) -> None:
        self.keyframes: list[TimestepKeyframe] = []
        self.keyframes.append(TimestepKeyframe.default())

    def add(self, keyframe: TimestepKeyframe) -> None:
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

    def get_index(self, index: int) -> TimestepKeyframe | None:
        try:
            return self.keyframes[index]
        except IndexError:
            return None
    
    def __getitem__(self, index) -> TimestepKeyframe:
        return self.keyframes[index]
    
    def is_empty(self) -> bool:
        return len(self.keyframes) == 0
    
    @classmethod
    def default(cls, keyframe: TimestepKeyframe) -> 'TimestepKeyframeGroup':
        group = cls()
        group.keyframes[0] = keyframe
        return group


# used to inject ControlNetAdvanced and T2IAdapterAdvanced control_merge function
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


class ControlNetAdvanced(ControlNet):
    def __init__(self, control_model, timestep_keyframes: TimestepKeyframeGroup, global_average_pooling=False, device=None):
        super().__init__(control_model=control_model, global_average_pooling=global_average_pooling, device=device)
        self.timestep_keyframes = timestep_keyframes if timestep_keyframes else TimestepKeyframeGroup()
        self.current_timestep_keyframe = self.timestep_keyframes.keyframes[0]
        # initialize weights
        self.weights = self.timestep_keyframes.keyframes[0].control_net_weights if self.timestep_keyframes.keyframes[0].control_net_weights else [1.0]*13
        # mask for which parts of controlnet output to keep
        self.cond_hint_mask = None
        # override control_merge
        self.control_merge = control_merge_inject.__get__(self, type(self))

    def get_control(self, x_noisy, t, cond, batched_number):
        # need to reference t and batched_number later
        self.t = t
        self.batched_number = batched_number
        # TODO: choose TimestepKeyframe based on t
        return super().get_control(x_noisy, t, cond, batched_number)

    def apply_advanced_strengths_and_masks(self, x, current_timestep_keyframe: TimestepKeyframe, batched_number: int):
        if current_timestep_keyframe.latent_keyframes is not None:
            # apply strengths, and get batch indeces to zero out
            # AKA latents that should not be influenced by ControlNet
            latent_count = x.size(0)//batched_number

            indeces_to_zero = set(range(latent_count))
            for keyframe in current_timestep_keyframe.latent_keyframes:
                if keyframe.batch_index in indeces_to_zero:
                    indeces_to_zero.remove(keyframe.batch_index)
                # apply strength for each batched cond/uncond
                for b in range(batched_number):
                    x[(latent_count*b)+keyframe.batch_index] *= keyframe.strength

            # zero them out by multiplying by zero
            for batch_index in indeces_to_zero:
                # apply zero for each batched cond/uncond
                for b in range(batched_number):
                    x[(latent_count*b)+batch_index] *= 0.0

    def copy(self):
        c = ControlNetAdvanced(self.control_model, self.timestep_keyframes, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out


class T2IAdapterAdvanced(T2IAdapter):
    def __init__(self, t2i_model, timestep_keyframes: TimestepKeyframeGroup, channels_in, device=None):
        super().__init__(t2i_model=t2i_model, channels_in=channels_in, device=device)
        self.timestep_keyframes = timestep_keyframes if timestep_keyframes else TimestepKeyframeGroup()
        self.current_timestep_keyframe = self.timestep_keyframes.keyframes[0]
        first_weight = self.timestep_keyframes.keyframes[0].t2i_adapter_weights if self.timestep_keyframes.get_index(0) else None
        self.weights = first_weight if first_weight else [1.0]*12
        # mask for which parts of controlnet output to keep
        self.cond_hint_mask = None
        # override control_merge
        self.control_merge = control_merge_inject.__get__(self, type(self))
    
    def get_control(self, x_noisy, t, cond, batched_number):
        # need to reference t and batched_number later
        self.t = t
        self.batched_number = batched_number
        # TODO: choose TimestepKeyframe based on t
        return super().get_control(x_noisy, t, cond, batched_number)

    def apply_advanced_strengths_and_masks(self, x, current_timestep_keyframe: TimestepKeyframe, batched_number: int):
        # For now, do nothing; need to figure out LatentKeyframe control is even possible for T2I Adapters
        return

    def copy(self):
        c = T2IAdapterAdvanced(self.t2i_model, self.timestep_keyframes, self.channels_in)
        self.copy_to(c)
        return c


def load_controlnet(ckpt_path, timestep_keyframe: TimestepKeyframeGroup=None, model=None):
    controlnet_data = utils.load_torch_file(ckpt_path, safe_load=True)
    if "lora_controlnet" in controlnet_data:
        return ControlLora(controlnet_data) # TODO: apply weights to ControlLora

    controlnet_config = None
    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        use_fp16 = model_management.should_use_fp16()
        controlnet_config = model_detection.unet_config_from_diffusers_unet(controlnet_data, use_fp16)
        diffusers_keys = utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            print("leftover keys:", leftover_keys)
        controlnet_data = new_sd

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        net = load_t2i_adapter(controlnet_data, timestep_keyframe)
        if net is None:
            print("error checkpoint does not contain controlnet or t2i adapter data", ckpt_path)
        return net

    if controlnet_config is None:
        use_fp16 = model_management.should_use_fp16()
        controlnet_config = model_detection.model_config_from_unet(controlnet_data, prefix, use_fp16).unet_config
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = cldm.ControlNet(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                model_management.load_models_gpu([model])
                model_sd = model.model_state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
            else:
                print("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
    print(missing, unexpected)

    if use_fp16:
        control_model = control_model.half()

    global_average_pooling = False
    if ckpt_path.endswith("_shuffle.pth") or ckpt_path.endswith("_shuffle.safetensors") or ckpt_path.endswith("_shuffle_fp16.safetensors"): #TODO: smarter way of enabling global_average_pooling
        global_average_pooling = True

    control = ControlNetAdvanced(control_model, timestep_keyframe, global_average_pooling=global_average_pooling)
    return control


def load_t2i_adapter(t2i_data, timestep_keyframes: TimestepKeyframeGroup=None):
    keys = t2i_data.keys()
    if 'adapter' in keys:
        t2i_data = t2i_data['adapter']
        keys = t2i_data.keys()
    if "body.0.in_conv.weight" in keys:
        cin = t2i_data['body.0.in_conv.weight'].shape[1]
        model_ad = t2i_adapter.adapter.Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif 'conv_in.weight' in keys:
        cin = t2i_data['conv_in.weight'].shape[1]
        channel = t2i_data['conv_in.weight'].shape[0]
        ksize = t2i_data['body.0.block2.weight'].shape[2]
        use_conv = False
        down_opts = list(filter(lambda a: a.endswith("down_opt.op.weight"), keys))
        if len(down_opts) > 0:
            use_conv = True
        xl = False
        if cin == 256 or cin == 768:
            xl = True
        model_ad = t2i_adapter.adapter.Adapter(cin=cin, channels=[channel, channel*2, channel*4, channel*4][:4], nums_rb=2, ksize=ksize, sk=True, use_conv=use_conv, xl=xl)
    else:
        return None
    missing, unexpected = model_ad.load_state_dict(t2i_data)
    if len(missing) > 0:
        print("t2i missing", missing)

    if len(unexpected) > 0:
        print("t2i unexpected", unexpected)
    
    return T2IAdapterAdvanced(model_ad, timestep_keyframes, model_ad.input_channels)
