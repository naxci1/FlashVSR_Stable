#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import shutil
import time
import math
import uuid
import re
import glob
import subprocess as sp
import itertools

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="FlashVSR+: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution.")
parser.add_argument("-i", "--input", type=str, required=True, help="Path to video file or folder of videos/images")
parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to save output video(s)")
parser.add_argument("-s", "--scale", type=int, default=4, help="Upscale factor, default=4")
parser.add_argument("-m", "--mode", type=str, default="tiny", choices=["tiny", "tiny-long", "full"], help="The type of pipeline to use, default=tiny")
parser.add_argument("--tiled-vae", action="store_true", help="Enable tile decoding for VAE")
parser.add_argument("--tiled-dit", action="store_true", help="Enable tile inference for DiT (for large resolutions)")
parser.add_argument("--tile-size", type=int, default=256, help="Chunk size of tile inference (input resolution), default=256")
parser.add_argument("--overlap", type=int, default=24, help="Overlap size of tile inference, default=24")
parser.add_argument("--unload-dit", action="store_true", help="Unload DiT before decoding to save VRAM")
parser.add_argument("--color-fix", action="store_true", help="Correct output video color")
parser.add_argument("--seed", type=int, default=0, help="Random Seed, default=0")
parser.add_argument("-t", "--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type for processing, default=bf16")
parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run FlashVSR on (e.g., 'cuda', 'cuda:0', 'mps', 'cpu')")
parser.add_argument("-f", "--fps", type=int, default=30, help="Output FPS (for image sequences only), default=30")
parser.add_argument("--qp", type=int, default=13, help="FFmpeg Quantization Parameter for av1_nvenc (0-51, lower is better). Default is 13.")
parser.add_argument("-a", "--attention", default="sage", choices=["sage", "block"], help="Attention mode, default=sage")
parser.add_argument("-np", "--negative-prompt", type=str, default="", help="Negative prompt to avoid concepts (e.g., 'glasses, blurry, ugly').")
parser.add_argument("--max-frames", type=int, default=None, help="Limit the number of frames to process from the start")
parser.add_argument("--batch-size", type=int, default=None, help="Process the video in chunks of this many frames")
args = parser.parse_args()

def log(message:str, message_type:str="normal"):
    if message_type == 'error': message = '\033[1;41m' + f"ERROR: {message}" + '\033[m'
    elif message_type == 'warning': message = '\033[1;31m' + f"WARNING: {message}" + '\033[m'
    elif message_type == 'finish': message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info': message = '\033[1;33m' + message + '\033[m'
    print(f"{message}")

try:
    from tqdm import tqdm
    import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    import imageio, ffmpeg, numpy as np, torch.nn.functional as F
    from PIL import Image
    from einops import rearrange
    from huggingface_hub import snapshot_download
    from decord import VideoReader, cpu
    from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
    from src.models import wan_video_dit
    from src.models.TCDecoder import build_tcdecoder
    from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj
except ImportError as e:
    log(f"A required library is missing: {e}. Please install all dependencies (including 'decord').", "error"); sys.exit(1)

root = os.path.dirname(os.path.abspath(__file__)); devices = get_device_list()

def model_download(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(root, "models", "FlashVSR")
    if not os.path.exists(model_dir): log(f"Downloading model '{model_name}' from Hugging Face...", 'info'); snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def is_ffmpeg_available():
    if shutil.which('ffmpeg') is None: log("FFmpeg not found! Please install FFmpeg to handle video audio.", "warning"); return False
    return True

def tensor2video(frames: torch.Tensor):
    if frames.ndim != 5: raise ValueError(f"Input tensor must be 5-dimensional (B, C, F, H, W), but got {frames.ndim} dimensions.")
    return (rearrange(frames.squeeze(0), "C F H W -> F H W C").float() + 1.0) / 2.0

def natural_key(name: str): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'); fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]; fs.sort(key=natural_key); return fs

def largest_8n1_leq(n): return 0 if n < 1 else ((n - 1) // 8) * 8 + 1

def is_video(path): return os.path.isfile(path) and path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))

def save_video(frames_gen, save_path, fps=30, qp_value=13):
    import itertools
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path): os.remove(save_path)
    
    sample_frame = next(frames_gen)
    H, W, C = sample_frame.shape
    frames_gen = itertools.chain([sample_frame], frames_gen)
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
        '-c:v', 'av1_nvenc', '-preset', 'p6', '-rc', 'constqp', '-qp', str(qp_value),
        '-pix_fmt', 'yuv420p', save_path
    ]
    
    process = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    
    for frame in frames_gen:
        frame_np = (frame.cpu().float().clamp(0, 1) * 255.0).numpy().astype(np.uint8)
        process.stdin.write(frame_np.tobytes())
    
    process.stdin.close()
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed: {stderr.decode('utf-8')}")
    log(f"High-quality video saved successfully to '{os.path.basename(save_path)}'", "finish")
    
def merge_video_with_audio(video_path, audio_source_path):
    if os.path.isdir(audio_source_path) or not is_ffmpeg_available(): log(f"Output video (without audio) saved to '{video_path}'", 'info'); return
    temp_path = video_path + ".temp-audio.mp4"
    try:
        if not any(s['codec_type'] == 'audio' for s in ffmpeg.probe(audio_source_path)['streams']): log(f"No audio stream in source. Final video saved to '{video_path}'", 'info'); return
        log("[FlashVSR] Copying audio tracks...", 'info'); os.rename(video_path, temp_path)
        ffmpeg.output(ffmpeg.input(temp_path)['v'], ffmpeg.input(audio_source_path)['a'], video_path, vcodec='copy', acodec='copy').run(overwrite_output=True, quiet=True)
        log(f"Final video with audio saved to '{video_path}'", 'finish')
    except ffmpeg.Error as e:
        log(f"Audio merge failed: {e.stderr.decode()}", 'warning')
        if os.path.exists(temp_path): os.rename(temp_path, video_path)
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError as e: log(f"Could not remove temporary file: {e}", 'warning')

def compute_scaled_and_target_dims(w0, h0, scale=4, multiple=128):
    sW, sH = w0 * scale, h0 * scale; tW = max(multiple, (sW // multiple) * multiple); tH = max(multiple, (sH // multiple) * multiple); return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor, scale, tW, tH):
    h0, w0, _ = frame_tensor.shape; tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    l, t = max(0, (sW - tW) // 2), max(0, (sH - tH) // 2)
    return upscaled_tensor[:, :, t:t + tH, l:l + tW].squeeze(0)

def prepare_tensors(path, dtype=torch.bfloat16):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0: raise FileNotFoundError(f"No images found in {path}")
        if args.max_frames is not None and args.max_frames > 0: paths0 = paths0[:args.max_frames]
        frames = [torch.from_numpy(np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0) for p in paths0]
        return torch.stack(frames, 0).pin_memory(), args.fps
    if is_video(path):
        try:
            vr = VideoReader(path, ctx=cpu(0)); fps = vr.get_avg_fps(); fps = round(fps)
            num_frames = min(len(vr), args.max_frames) if args.max_frames is not None and args.max_frames > 0 else len(vr)
            frames = torch.from_numpy(vr.get_batch(range(num_frames)).asnumpy().astype(np.float32) / 255.0)
            if not frames.shape[0] > 0: raise RuntimeError(f"Could not read any frames from {path}")
            return frames.pin_memory(), fps
        except Exception as e: raise RuntimeError(f"Failed to read video file {path} with decord: {e}")
    raise ValueError(f"Unsupported input format: {path}")

def get_input_params(image_tensor, scale):
    N0, h0, w0, _ = image_tensor.shape; _, _, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale)
    F = largest_8n1_leq(N0 + 4)
    if F == 0: raise RuntimeError(f"Not enough frames. Need at least 5 frames, but got {N0}.")
    return tH, tW, F

def input_tensor_generator(image_tensor, device, scale=4, dtype=torch.bfloat16):
    N0, _, _, _ = image_tensor.shape; tH, tW, F = get_input_params(image_tensor, scale)
    for i in range(F):
        frame_idx = min(i, N0 - 1); frame_slice = image_tensor[frame_idx]
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH)
        yield (tensor_chw * 2.0 - 1.0).to('cpu').to(dtype)

def prepare_input_tensor(image_tensor, device, scale=4, dtype=torch.bfloat16):
    N0 = image_tensor.shape[0]; tH, tW, F = get_input_params(image_tensor, scale)
    frames = [tensor_upscale_then_center_crop(image_tensor[min(i, N0 - 1)], scale=scale, tW=tW, tH=tH) for i in range(F)]
    vid_stacked = torch.stack(frames, 0)
    vid_final = (vid_stacked * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
    clean_vram(); return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords, stride = [], tile_size - overlap; num_rows, num_cols = math.ceil((height - overlap) / stride), math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, x1 = r * stride, c * stride; y2, x2 = min(y1 + tile_size, height), min(x1 + tile_size, width)
            if y2 - y1 < tile_size: y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size: x1 = max(0, x2 - tile_size)
            coords.append((x1, y1, x2, y2))
    return coords

def create_feather_mask(size, overlap, device):
    H, W = size; mask = torch.ones(1, 1, H, W, device=device)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap, device=device); mask[..., :, :overlap] *= ramp.view(1, 1, 1, -1); mask[..., :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)
        mask[..., :overlap, :] *= ramp.view(1, 1, -1, 1); mask[..., -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
    return mask

def init_pipeline(mode, device, dtype):
    model_download(); model_path = os.path.join(root, "models", "FlashVSR")
    required_files = ["diffusion_pytorch_model_streaming_dmd.safetensors", "Wan2.1_VAE.pth", "LQ_proj_in.ckpt", "TCDecoder.ckpt"]
    for fname in required_files:
        if not os.path.exists(os.path.join(model_path, fname)): raise RuntimeError(f'Model file "{fname}" not found')
    ckpt_path, vae_path, lq_path, tcd_path = [os.path.join(model_path, f) for f in required_files]
    prompt_path = os.path.join(root, "models", "posi_prompt.pth")
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path]); pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
    else:
        mm.load_models([ckpt_path]); pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device) if mode == "tiny-long" else FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device=device, dtype=dtype, new_latent_channels=16+768)
        pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False); pipe.TCDecoder.clean_mem()
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    if os.path.exists(lq_path): pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.to(device, dtype=dtype); pipe.enable_vram_management(); pipe.init_cross_kv(prompt_path=prompt_path)
    log("Applying torch.compile() for maximum performance...", "info")
    try:
        if hasattr(pipe, 'dit') and pipe.dit is not None: pipe.dit = torch.compile(pipe.dit, mode="max-autotune", fullgraph=True)
        if hasattr(pipe, 'vae') and pipe.vae is not None: pipe.vae = torch.compile(pipe.vae, mode="max-autotune", fullgraph=True)
        if hasattr(pipe, 'TCDecoder') and pipe.TCDecoder is not None: pipe.TCDecoder = torch.compile(pipe.TCDecoder, mode="max-autotune", fullgraph=True)
        log("torch.compile() applied successfully.", "finish")
    except Exception as e: log(f"torch.compile() failed: {e}. Running without this optimization.", "warning")
    log("Moving models to GPU...", "info")
    if hasattr(pipe, 'dit') and pipe.dit is not None: pipe.dit.to(device)
    if hasattr(pipe, 'vae') and pipe.vae is not None: pipe.vae.to(device)
    if hasattr(pipe, 'TCDecoder') and pipe.TCDecoder is not None: pipe.TCDecoder.to(device)
    return pipe

def process_frames(
    frames, fps, mode, scale, color_fix, tiled_vae, tiled_dit,
    tile_size, tile_overlap, unload_dit, dtype, seed=0, device="auto",
    qp=13, negative_prompt="", output=None, pipe=None
):
    _device = device if device != "auto" else "cuda"
    
    if frames.shape[0] < 5:
        log(f"A chunk must have at least 5 frames, got {frames.shape[0]}. Skipping.", "warning")
        return None, fps

    if pipe is None:
        pipe = init_pipeline(mode, _device, dtype)

    is_long_mode = (mode == "tiny-long")

    if tiled_dit and not is_long_mode:
        N, H, W, C = frames.shape
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        num_output_frames = largest_8n1_leq(N + 4) - 4

        if num_output_frames <= 0:
            log("[FlashVSR] Not enough frames for processing this chunk. Skipping.", "warning")
            return None, fps

        log(f"[FlashVSR] Using tiled processing for {len(tile_coords)} tiles. Storing result on CPU to save VRAM.", "info")
        final_output_canvas = torch.zeros((num_output_frames, H * scale, W * scale, C), dtype=torch.float32, device='cpu')
        weight_sum_canvas = torch.zeros_like(final_output_canvas, device='cpu')

        for idx, (x1, y1, x2, y2) in enumerate(tile_coords, 1):
            log(f"[Tile {idx}/{len(tile_coords)}] Extracting and preparing input...", "info")
            input_tile_gpu = frames[:, y1:y2, x1:x2, :]
            LQ_tile, th, tw, F = prepare_input_tensor(input_tile_gpu, _device, scale, dtype)

            log(f"[Tile {idx}/{len(tile_coords)}] Running pipeline...", "info")
            output_tile_gpu = pipe(
                prompt="",
                negative_prompt=negative_prompt,
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=seed,
                tiled=tiled_vae,
                LQ_video=LQ_tile,
                num_frames=F,
                height=th,
                width=tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=2*768*1280/(th*tw),
                local_range=11,
                color_fix=color_fix,
                unload_dit=unload_dit,
                fps=fps,
                output_path=None,
                tiled_dit=True
            )

            if output_tile_gpu is None:
                continue
            if output_tile_gpu.ndim == 4:
                output_tile_gpu = output_tile_gpu.unsqueeze(0)

            processed_tile_gpu = tensor2video(output_tile_gpu)
            if processed_tile_gpu.shape[0] > num_output_frames:
                processed_tile_gpu = processed_tile_gpu[:num_output_frames]

            mask_gpu = create_feather_mask(
                processed_tile_gpu.shape[1:3],
                tile_overlap * scale,
                device=_device
            ).permute(0, 2, 3, 1)

            out_x1, out_y1 = x1 * scale, y1 * scale
            final_output_canvas[:, out_y1:out_y1+processed_tile_gpu.shape[1],
                                out_x1:out_x1+processed_tile_gpu.shape[2], :] += (processed_tile_gpu * mask_gpu).cpu()
            weight_sum_canvas[:, out_y1:out_y1+processed_tile_gpu.shape[1],
                              out_x1:out_x1+processed_tile_gpu.shape[2], :] += mask_gpu.cpu()

            del LQ_tile, input_tile_gpu, output_tile_gpu, processed_tile_gpu, mask_gpu
            clean_vram()
            torch.cuda.empty_cache()

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas

    else:
        if is_long_mode:
            LQ, th, tw, F = input_tensor_generator(frames, _device, scale, dtype), *get_input_params(frames, scale)
        else:
            LQ, th, tw, F = prepare_input_tensor(frames, _device, scale, dtype)

        log(f"[FlashVSR] Processing {frames.shape[0]} frames on {_device}...", "info")
        video = pipe(
            prompt="", negative_prompt=negative_prompt, cfg_scale=1.0, num_inference_steps=1, seed=seed,
            tiled=tiled_vae, LQ_video=LQ, num_frames=F, height=th, width=tw,
            is_full_block=False, if_buffer=True, topk_ratio=2*768*1280/(th*tw),
            local_range=11, color_fix=color_fix, unload_dit=unload_dit, fps=fps,
            output_path=None, tiled_dit=tiled_dit
        )

        if video.ndim == 5:
            final_output = tensor2video(video)
        elif video.ndim == 4:
            final_output = video
        else:
            final_output = None

        del video, LQ
        clean_vram()

    return final_output, fps

if __name__ == "__main__":
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.bfloat16); wan_video_dit.USE_BLOCK_ATTN = (args.attention != "sage")
    try: model_download()
    except Exception as e: log(f"Failed to download models: {e}", "error"); sys.exit(1)
    if os.path.isdir(args.input):
        video_list = [os.path.join(args.input, f) for f in os.listdir(args.input) if is_video(os.path.join(args.input, f))]
        video_list.sort(key=natural_key)
    else: video_list = [args.input] if os.path.exists(args.input) else []
    if not video_list: log(f"No video files found in '{args.input}'!", "error"); sys.exit(1)
    os.makedirs(args.output_folder, exist_ok=True); log(f"[FlashVSR] Found {len(video_list)} video(s) to process.", "info")
    for vid_path in video_list:
        overall_start_time = time.time(); base_name, _ = os.path.splitext(os.path.basename(vid_path))
        output_base_name = f"{base_name}_{args.scale}x"
        output_filename = f"{output_base_name}.mp4"
        final_output_path = os.path.join(args.output_folder, output_filename)
        counter = 1
        while os.path.exists(final_output_path):
            output_filename = f"{output_base_name}_{counter}.mp4"
            final_output_path = os.path.join(args.output_folder, output_filename)
            counter += 1
        log(f"==> Processing: {os.path.basename(vid_path)} | Output: {os.path.basename(final_output_path)}", "info")
        try:
            all_frames_cpu, result_fps = prepare_tensors(vid_path, dtype); total_frames = all_frames_cpu.shape[0]
            log(f"Successfully loaded {total_frames} frames into pinned memory.", "finish")
            processed_chunks_in_ram = []; batch_size = args.batch_size if args.batch_size and args.batch_size > 0 else total_frames
            num_chunks = math.ceil(total_frames / batch_size)
            pipe = init_pipeline(args.mode, args.device, dtype); stream = torch.cuda.Stream()
            for i in range(num_chunks):
                start_frame, end_frame = i * batch_size, min((i + 1) * batch_size, total_frames)
                log(f"--- Processing Chunk {i+1}/{num_chunks} (Frames {start_frame}-{end_frame}) ---", "info")
                with torch.cuda.stream(stream):
                    frame_chunk_cpu = all_frames_cpu[start_frame:end_frame]
                    frame_chunk_gpu = frame_chunk_cpu.to(device=args.device, dtype=dtype, non_blocking=True)
                    chunk_result, _ = process_frames(frames=frame_chunk_gpu, fps=result_fps, mode=args.mode, scale=args.scale, color_fix=args.color_fix, tiled_vae=args.tiled_vae, tiled_dit=args.tiled_dit, tile_size=args.tile_size, tile_overlap=args.overlap, unload_dit=args.unload_dit, dtype=dtype, seed=args.seed, device=args.device, qp=args.qp, negative_prompt=args.negative_prompt, output=None, pipe=pipe)
                if chunk_result is not None: processed_chunks_in_ram.append(chunk_result.cpu())
                clean_vram()
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if processed_chunks_in_ram:
                log("All chunks processed. Concatenating and saving...", "info")
                final_tensor = torch.cat(processed_chunks_in_ram, dim=0)
                save_video(iter(final_tensor), final_output_path, fps=int(result_fps), qp_value=args.qp)
                merge_video_with_audio(final_output_path, vid_path)
            else: log("No chunks resulted in output.", "warning")
            elapsed = time.time() - overall_start_time
            log(f"<== Finished '{os.path.basename(vid_path)}' in {elapsed:.2f}s ({total_frames / elapsed:.2f} FPS).", "finish")
        except Exception as e:
            log(f"Error processing '{os.path.basename(vid_path)}': {e}", "error")
            import traceback; traceback.print_exc()
    log("All videos processed successfully!", "finish")