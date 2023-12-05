import torch
from comfy.model_management import get_torch_device, soft_empty_cache
import bisect
from typing import List
import numpy as np
import ast
import typing
import pathlib
import einops
import traceback
import os
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
import yaml


BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
]

config_path = os.path.join(os.path.dirname(__file__), "./config.yaml")
if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    raise Exception("config.yaml file is neccessary, plz recreate the config file by downloading it from https://github.com/Fannovel16/ComfyUI-Frame-Interpolation")
DEVICE = get_torch_device()

def get_ckpt_container_path(model_type):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), config["ckpts_path"], model_type))


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):

    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    file_name = os.path.basename(parts.path)
    if file_name is not None:
        file_name = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

class InterpolationStateList():

    def __init__(self, frame_indices: typing.List[int], is_skip_list: bool):
        self.frame_indices = frame_indices
        self.is_skip_list = is_skip_list
        
    def is_frame_skipped(self, frame_index):
        is_frame_in_list = frame_index in self.frame_indices
        return self.is_skip_list and is_frame_in_list or not self.is_skip_list and not is_frame_in_list
    

def load_file_from_github_release(model_type, ckpt_name):
    error_strs = []
    for i, base_model_download_url in enumerate(BASE_MODEL_DOWNLOAD_URLS):
        try:
            return load_file_from_url(base_model_download_url + ckpt_name, get_ckpt_container_path(model_type))
        except Exception:
            traceback_str = traceback.format_exc()
            if i < len(BASE_MODEL_DOWNLOAD_URLS) - 1:
                print("Failed! Trying another endpoint.")
            error_strs.append(f"Error when downloading from: {base_model_download_url + ckpt_name}\n\n{traceback_str}")

    error_str = '\n\n'.join(error_strs)
    raise Exception(f"Tried all GitHub base urls to download {ckpt_name} but no suceess. Below is the error log:\n\n{error_str}")


def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()


MODEL_TYPE = pathlib.Path(__file__).parent.name
DEVICE = get_torch_device()

def inference(model, img_batch_1, img_batch_2, inter_frames):
    results = [
        img_batch_1,
        img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i].to(DEVICE)
        x1 = results[end_i].to(DEVICE)
        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).float())
        del remains[step]

    return [tensor.flip(0) for tensor in results]


def film_interpolation(    
    frames: torch.Tensor,                
    frame_counts: List[int] = [0,5,32,40],
    buffer: int = 10):        

    frame_counts = sorted(frame_counts)

    max_gap = max(b-a for a, b in zip(frame_counts[:-1], frame_counts[1:]))
            
    model_path = load_file_from_github_release(MODEL_TYPE, "film_net_fp32.pt")
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    model = model.to(DEVICE)

    frames = preprocess_frames(frames)
    number_of_frames_processed_since_last_cleared_cuda_cache = 0
    clear_cache_after_n_frames = 10  # Example value

    # Generate buffer frames and attach them to the beginning of output_frames
    first_frame = frames[0].unsqueeze(0)
    buffer_frames = [first_frame] * buffer
    output_frames = buffer_frames


    for frame_itr in range(len(frames) - 1):


        frame_0 = frames[frame_itr:frame_itr+1].to(DEVICE)
        frame_1 = frames[frame_itr+1:frame_itr+2].to(DEVICE)
        frame_output = []
        result = inference(model, frame_0, frame_1, max_gap - 1)

        # Find the current frame's position in the frame_counts list        
        current_frame = frame_counts[frame_itr]
        next_frame = frame_counts[frame_itr + 1] - 1
        current_gap = next_frame - current_frame
        # Determine the number of frames to drop based on the difference between the max gap and the current gap
        frames_to_drop = max_gap - current_gap
        
        frame_output = result[:-1]
        
        # frames_to_drop = max_gap - 1 - len(frame_output)
        if frames_to_drop > 0:
            if frames_to_drop >= len(frame_output):
                raise ValueError("Number of frames to drop is greater than or equal to total number of frames in the batch.")
                            
            drop_interval = len(frame_output) / float(frames_to_drop)
            result = []
            next_drop = drop_interval

            for i, frame in enumerate(frame_output):
                if i >= next_drop:
                    next_drop += drop_interval
                else:
                    result.append(frame)

            frame_output = result  # Update frame_output with only the undropped frames

        # Detach and move to CPU all frames, whether or not any were dropped
        frame_output = [frame.detach().cpu() for frame in frame_output]

        # Append the processed (and potentially dropped) frames to the main list
        output_frames.extend(frame_output)
                    
        number_of_frames_processed_since_last_cleared_cuda_cache += 1
        if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
            print("Comfy-VFI: Clearing cache...")
            soft_empty_cache()
            number_of_frames_processed_since_last_cleared_cuda_cache = 0
            print("Comfy-VFI: Done cache clearing")
        
    output_frames.append(frames[-1:])
    out = torch.cat(output_frames, dim=0)

    # clear cache for courtesy
    print("Comfy-VFI: Final clearing cache...")
    soft_empty_cache()
    print("Comfy-VFI: Done cache clearing")
            

    return (postprocess_frames(out), )