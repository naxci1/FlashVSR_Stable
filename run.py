#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

parser = argparse.ArgumentParser(description="FlashVSR+: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution.")
parser.add_argument("-i", "--input", type=str, help="Path to video file or folder of images")
parser.add_argument("-s", "--scale", type=int, default=4, help="Upscale factor, default=4")
parser.add_argument("-m", "--mode", type=str, default="tiny", choices=["tiny", "full"], help="The type of pipeline to use, default=tiny")
parser.add_argument("--tiled-vae", action="store_true", help="Enable tile decoding")
parser.add_argument("--tiled-dit", action="store_true", help="Enable tile inference")
parser.add_argument("--tile-size", type=int, default=256, help="Chunk size of tile inference, default=256")
parser.add_argument("--overlap", type=int, default=24, help="Overlap size of tile inference, default=24")
parser.add_argument("--unload-dit", action="store_true", help="Unload DiT before decoding")
parser.add_argument("--color-fix", action="store_true", help="Correct output video color")
parser.add_argument("--seed", type=int, default=0, help="Random Seed, default=0")
parser.add_argument("-t", "--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type for processing, default=bf16")
parser.add_argument("-d", "--device", type=str, default="auto", help="Device to run FlashVSR")
parser.add_argument("-f", "--fps", type=int, default=0, help="Output frame rate (0=same as input), default=0")
parser.add_argument("-q", "--quality", type=int, default=6, help="Output video quality, default=6")
parser.add_argument("output_folder", type=str, help="Path to save output video")
args = parser.parse_args()

import os
import re
import math
import torch
import shutil
import imageio
import ffmpeg
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from huggingface_hub import snapshot_download
from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj

root = os.path.dirname(os.path.abspath(__file__))
devices = get_device_list()

def log(message:str, message_type:str='info'):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"{message}")

def model_downlod(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(root, "models", "FlashVSR")
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def is_ffmpeg_available():
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        log("[FlashVSR] FFmpeg not found!", message_type="warning")
        log("Please install FFmpeg and ensure it is in your system's PATH.")
        log("- Windows: Download from https://www.ffmpeg.org/download.html and add the 'bin' directory to PATH.")
        log("- macOS (via Homebrew): brew install ffmpeg")
        log("- Linux (Ubuntu/Debian): sudo apt-get install ffmpeg")
        return False
    return True

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def tensor2images(frames: torch.Tensor):
    frames_np = (frames.cpu().float() * 255.0).clip(0, 255).numpy().astype(np.uint8)
    image_list = [Image.fromarray(frame) for frame in frames_np]
    return image_list

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path): 
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"[FlashVSR] Saving video"):
        w.append_data(np.array(f))
    w.close()

def merge_video_with_audio(video_path, audio_source_path, output_path):
    if os.path.isdir(video_path):
        os.rename(video_path, output_path)
        log(f"[FlashVSR] Output video saved to '{output_path}'", message_type='info')
        return
    
    if not is_ffmpeg_available():
        os.rename(video_path, output_path)
        log(f"[FlashVSR] Output video saved to '{output_path}'", message_type='info')
        return
    
    try:
        probe = ffmpeg.probe(audio_source_path)
        audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
        if not audio_streams:
            log(f"[FlashVSR] Output video saved to '{output_path}'", message_type='info')
            os.rename(video_path, output_path)
            return
        
        input_video = ffmpeg.input(video_path)['v']
        input_audio = ffmpeg.input(audio_source_path)['a']
        output_ffmpeg = ffmpeg.output(
            input_video, input_audio, output_path,
            vcodec='copy', acodec='copy'
        ).run(overwrite_output=True, quiet=True)
        log(f"[FlashVSR] Output video saved to '{output_path}'", message_type='info')
    except ffmpeg.Error as e:
        os.rename(video_path, output_path)
        print("[ERROR] FFmpeg error during merge:", e.stderr.decode() if e.stderr else "Unknown error")
        log(f"[FlashVSR] Audio merge failed. A silent video has been saved to '{output_path}'.", message_type='warning')
        
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as e:
                lgo(f"[FlashVSR] Could not remove temporary file '{video_path}': {e}", message_type='error')
    
def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")
        
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    
    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
    
    return cropped_tensor.squeeze(0)

def prepare_tensors(path: str, dtype=torch.bfloat16):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
            
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        
        frames = []
        for p in paths0:
            with Image.open(p).convert('RGB') as img:
                img_np = np.array(img).astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img_np).to(dtype))
                
        vid = torch.stack(frames, 0)
        fps = 30
        return vid, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        meta = {}
        try:
            meta = rdr.get_meta_data()
            first_frame = rdr.get_data(0)
            h0, w0, _ = first_frame.shape
        except Exception:
            first_frame = rdr.get_data(0)
            h0, w0, _ = first_frame.shape
            
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30
        
        total = meta.get('nframes', rdr.count_frames())
        if total is None or total <= 0 :
             total = len([_ for _ in rdr])
             rdr = imageio.get_reader(path)
            
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        frames = []
        try:
            for frame_data in rdr:
                frame_np = frame_data.astype(np.float32) / 255.0
                frames.append(torch.from_numpy(frame_np).to(dtype))
        finally:
            try:
                rdr.close()
            except Exception:
                pass
        vid = torch.stack(frames, 0)
        return vid, fps
    
    raise ValueError(f"Unsupported input: {path}")

def prepare_input_tensor(image_tensor: torch.Tensor, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
        
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx]
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        tensor_out = tensor_out.to(dtype)
        frames.append(tensor_out)
        
    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def init_pipeline(mode, device, dtype):
    model_downlod()
    model_path = os.path.join(root, "models", "FlashVSR")
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist! Please save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist! Please save it to "{model_path}"')
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_path):
        raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist! Please save it to "{model_path}"')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist! Please save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist! Please save it to "{model_path}"')
    prompt_path = os.path.join(root, "models", "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()
        
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    if os.path.exists(lq_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    
    return pipe

def main(input, mode, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, dtype, sparse_ratio=2, kv_ratio=3, local_range=11, seed=0, device="auto"):
    _device = device
    if device == "auto":
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else device
    if _device == "auto" or _device not in devices:
        raise RuntimeError("No devices found to run FlashVSR!")
    if _device.startswith("cuda"):
        torch.cuda.set_device(_device)
        
    if tiled_dit and (tile_overlap > tile_size / 2):
        raise ValueError('The "tile_overlap" must be less than half of "tile_size"!')
    
    pipe = init_pipeline(mode, _device, dtype)
    frames, fps = prepare_tensors(input, dtype=dtype)
    
    if frames.shape[0] < 21:
        raise ValueError(f"Number of frames must be at least 21, got {frames.shape[0]}")
    
    if tiled_dit:
        N, H, W, C = frames.shape
        num_aligned_frames = largest_8n1_leq(N + 4) - 4
        
        final_output_canvas = torch.zeros(
            (num_aligned_frames, H * scale, W * scale, C), 
            dtype=torch.float32, 
            device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        latent_tiles_cpu = []
        
        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
            log(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: coords ({x1},{y1}) to ({x2},{y2})", message_type='info')
            input_tile = frames[:, y1:y2, x1:x2, :]
            
            _tile = input_tile.to(_device)
            LQ_tile, th, tw, F = prepare_input_tensor(_tile, scale=scale, dtype=dtype)
            del _tile
            clean_vram()
            
            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, unload_dit=unload_dit
            )
            
            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
            
            mask_nchw = create_feather_mask(
                (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                tile_overlap * scale
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
            out_x1, out_y1 = x1 * scale, y1 * scale
            
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            
            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()
            
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        log(f"[FlashVSR] Processing {frames.shape[0]} frames...", message_type='info')
        
        _frames = frames.to(_device)
        LQ, th, tw, F = prepare_input_tensor(_frames, scale=scale, dtype=dtype)
        
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix = color_fix, unload_dit=unload_dit
        )
        
        final_output = tensor2video(video)
        
        del pipe, video, LQ
        clean_vram()
    
    return final_output, fps

if __name__ == "__main__":
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        dtype = dtype_map[args.dtype]
    except:
        dtype = torch.bfloat16
    
    model_downlod()
    result, fps = main(args.input, args.mode, args.scale, args.color_fix, args.tiled_vae, args.tiled_dit,
        args.tile_size, args.overlap, args.unload_dit, dtype, seed=args.seed, device=args.device)
    video = tensor2images(result)
    
    _fps = args.fps if args.fps != 0 else fps
    name = os.path.basename(args.input.rstrip('/'))
    temp = os.path.join(args.output_folder, f"FlashVSR_{args.mode}_{name.split('.')[0]}_{args.seed}_temp.mp4")
    final = os.path.join(args.output_folder, f"FlashVSR_{args.mode}_{name.split('.')[0]}_{args.seed}.mp4")
    save_video(video, temp, fps=_fps, quality=args.quality)
    merge_video_with_audio(temp, args.input, final)
    log("[FlashVSR] Done.", message_type='finish')