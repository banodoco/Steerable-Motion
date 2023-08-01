import sys
import os


import torch
import contextlib
import copy
import inspect


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from comfy.cldm import cldm
from comfy.t2i_adapter import adapter

from comfy.sd import ControlBase, broadcast_image_to
import comfy.utils as utils
import comfy.model_management as model_management
import comfy.model_detection as model_detection


ControlNetWeightsType = list[float]
T2IAdapterWeightsType = list[float]


# Copied from comfy.sd, weights modified
class ControlNetAdvanced(ControlBase):
    def __init__(self, control_model, weights: ControlNetWeightsType, global_average_pooling=False, device=None):
        super().__init__(device)
        self.control_model = control_model
        self.weights = weights
        self.global_average_pooling = global_average_pooling

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return {}

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(self.control_model.dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        if self.control_model.dtype == torch.float16:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(model_management.get_autocast_device(self.device)):
            self.control_model = model_management.load_if_low_vram(self.control_model)
            context = torch.cat(cond['c_crossattn'], 1)
            y = cond.get('c_adm', None)
            control = self.control_model(x=x_noisy, hint=self.cond_hint, timesteps=t, context=context, y=y)
            self.control_model = model_management.unload_if_low_vram(self.control_model)
        out = {'middle':[], 'output': []}
        autocast_enabled = torch.is_autocast_enabled()

        for i in range(len(control)):
            if i == (len(control) - 1):
                key = 'middle'
                index = 0
            else:
                key = 'output'
                index = i
            x = control[i]
            if self.global_average_pooling:
                x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

            #multiplier = 1#0.825**float(12-i)
            #print(f"$$$ multiplier: {multiplier}")
            x *= self.strength*self.weights[i]
            if x.dtype != output_dtype and not autocast_enabled:
                x = x.to(output_dtype)

            if control_prev is not None and key in control_prev:
                prev = control_prev[key][index]
                if prev is not None:
                    x += prev
            out[key].append(x)
        if control_prev is not None and 'input' in control_prev:
            out['input'] = control_prev['input']
        return out

    def copy(self):
        c = ControlNetAdvanced(self.control_model, self.weights, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model)
        return out


def load_controlnet(ckpt_path, control_net_weights: ControlNetWeightsType=None, t2i_adapter_weights: T2IAdapterWeightsType=None, model=None):
    controlnet_data = utils.load_torch_file(ckpt_path, safe_load=True)

    controlnet_config = None
    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        use_fp16 = model_management.should_use_fp16()
        controlnet_config = model_detection.model_config_from_diffusers_unet(controlnet_data, use_fp16).unet_config
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
        net = load_t2i_adapter(controlnet_data, t2i_adapter_weights)
        if net is None:
            print("error checkpoint does not contain controlnet or t2i adapter data", ckpt_path)
        return net

    if controlnet_config is None:
        use_fp16 = model_management.should_use_fp16()
        controlnet_config = model_detection.model_config_from_unet(controlnet_data, prefix, use_fp16).unet_config
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = 3
    control_model = cldm.ControlNet(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                m = model.patch_model()
                model_sd = m.state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
                model.unpatch_model()
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

    control = ControlNetAdvanced(control_model, control_net_weights, global_average_pooling=global_average_pooling)
    return control


# Copied from comfy.sd, weights modified
class T2IAdapterAdvanced(ControlBase):
    def __init__(self, t2i_model, weights: T2IAdapterWeightsType, channels_in, device=None):
        super().__init__(device)
        self.t2i_model = t2i_model
        self.weights = weights if weights else [1.0]*4
        self.channels_in = channels_in
        self.control_input = None

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return {}

        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = None
            self.cond_hint = utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").float().to(self.device)
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)
        if self.control_input is None:
            self.t2i_model.to(self.device)
            self.control_input = self.t2i_model(self.cond_hint)
            self.t2i_model.cpu()

        output_dtype = x_noisy.dtype
        out = {'input':[]}

        autocast_enabled = torch.is_autocast_enabled()
        print(f"$$$$ t2i control_input len: {len(self.control_input)}")
        for i in range(len(self.control_input)):
            key = 'input'
            x = self.control_input[i] * self.strength
            if x.dtype != output_dtype and not autocast_enabled:
                x = x.to(output_dtype)

            if control_prev is not None and key in control_prev:
                index = len(control_prev[key]) - i * 3 - 3
                prev = control_prev[key][index]
                if prev is not None:
                    x += prev
            out[key].insert(0, None)
            out[key].insert(0, None)
            out[key].insert(0, x)

        if control_prev is not None and 'input' in control_prev:
            for i in range(len(out['input'])):
                if out['input'][i] is None:
                    out['input'][i] = control_prev['input'][i]
        if control_prev is not None and 'middle' in control_prev:
            out['middle'] = control_prev['middle']
        if control_prev is not None and 'output' in control_prev:
            out['output'] = control_prev['output']
        return out

    def copy(self):
        c = T2IAdapterAdvanced(self.t2i_model, self.weights, self.channels_in)
        self.copy_to(c)
        return c


def load_t2i_adapter(t2i_data, weights: T2IAdapterWeightsType=None):
    keys = t2i_data.keys()
    if 'adapter' in keys:
        t2i_data = t2i_data['adapter']
        keys = t2i_data.keys()
    if "body.0.in_conv.weight" in keys:
        cin = t2i_data['body.0.in_conv.weight'].shape[1]
        model_ad = adapter.Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif 'conv_in.weight' in keys:
        cin = t2i_data['conv_in.weight'].shape[1]
        channel = t2i_data['conv_in.weight'].shape[0]
        ksize = t2i_data['body.0.block2.weight'].shape[2]
        use_conv = False
        down_opts = list(filter(lambda a: a.endswith("down_opt.op.weight"), keys))
        if len(down_opts) > 0:
            use_conv = True
        model_ad = adapter.Adapter(cin=cin, channels=[channel, channel*2, channel*4, channel*4][:4], nums_rb=2, ksize=ksize, sk=True, use_conv=use_conv)
    else:
        return None
    model_ad.load_state_dict(t2i_data)
    return T2IAdapterAdvanced(model_ad, weights, cin // 64)
