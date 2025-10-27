import sys
import argparse
parser = argparse.ArgumentParser(description="FlashVSR+ WebUI")
parser.add_argument("--listen", action="store_true", help="Allow LAN access")
parser.add_argument("--port", type=int, default=7860, help="Service Port")
args = parser.parse_args()

import gradio as gr
import os
import re
import math
import uuid
import torch
import shutil
import imageio
import ffmpeg
import numpy as np
import torch.nn.functional as F
import random
import time
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from huggingface_hub import snapshot_download

from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from src.models import wan_video_dit
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
TEMP_DIR = os.path.join(ROOT_DIR, "_temp")
os.environ['GRADIO_TEMP_DIR'] = TEMP_DIR

def log(message:str, message_type:str="normal"):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

def dummy_tqdm(iterable, *args, **kwargs):
    return iterable

def model_downlod(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(ROOT_DIR, "models", "FlashVSR")
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path): 
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def save_video(frames, save_path, fps=30, quality=5, progress_desc="Saving video..."):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames_np = (frames.cpu().float() * 255.0).clip(0, 255).numpy().astype(np.uint8)
    with imageio.get_writer(save_path, fps=fps, quality=quality) as writer:
        for frame_np in tqdm(frames_np, desc=f"[FlashVSR] {progress_desc}"):
            writer.append_data(frame_np)

def prepare_tensors(path: str, dtype=torch.bfloat16):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0: raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0: w0, h0 = _img0.size
        frames = [torch.from_numpy(np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0).to(dtype) for p in tqdm(paths0, desc="Loading images")]
        return torch.stack(frames, 0), 30
    if is_video(path):
        with imageio.get_reader(path) as rdr:
            meta = rdr.get_meta_data()
            fps = meta.get('fps', 30)
            frames = [torch.from_numpy(frame_data.astype(np.float32) / 255.0).to(dtype) for frame_data in tqdm(rdr, desc="Loading video frames")]
        return torch.stack(frames, 0), fps
    raise ValueError(f"Unsupported input: {path}")

def get_input_params(image_tensor, scale):
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH, tW, tH = w0 * scale, h0 * scale, max(multiple, (w0 * scale // multiple) * multiple), max(multiple, (h0 * scale // multiple) * multiple)
    F = largest_8n1_leq(N0 + 4)
    if F == 0: raise RuntimeError(f"Not enough frames. Got {N0 + 4}.")
    return tH, tW, F

def input_tensor_generator(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, Fs = get_input_params(image_tensor, scale)
    for i in range(Fs):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
        upscaled_tensor = F.interpolate(tensor_bchw, size=(h0 * scale, w0 * scale), mode='bicubic', align_corners=False)
        l, t = max(0, (w0 * scale - tW) // 2), max(0, (h0 * scale - tH) // 2)
        cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
        tensor_out = (cropped_tensor.squeeze(0) * 2.0 - 1.0)
        yield tensor_out.to('cpu').to(dtype)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, Fs = get_input_params(image_tensor, scale)
    frames = []
    for i in range(Fs):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
        upscaled_tensor = F.interpolate(tensor_bchw, size=(h0 * scale, w0 * scale), mode='bicubic', align_corners=False)
        l, t = max(0, (w0 * scale - tW) // 2), max(0, (h0 * scale - tH) // 2)
        cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
        tensor_out = (cropped_tensor.squeeze(0) * 2.0 - 1.0).to('cpu').to(dtype)
        frames.append(tensor_out)
    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    clean_vram()
    return vid_final, tH, tW, Fs

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    stride = tile_size - overlap
    num_rows, num_cols = math.ceil((height - overlap) / stride), math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + tile_size, height), min(x1 + tile_size, width)
            if y2 - y1 < tile_size: y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size: x1 = max(0, x2 - tile_size)
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

def stitch_video_tiles(
    tile_paths, 
    tile_coords, 
    final_dims, 
    scale, 
    overlap, 
    output_path, 
    fps, 
    quality, 
    cleanup=True,
    chunk_size=40
):
    if not tile_paths:
        log("No tile videos found to stitch.", message_type='error')
        return
    
    final_W, final_H = final_dims
    
    readers = [imageio.get_reader(p) for p in tile_paths]
    
    try:
        num_frames = readers[0].count_frames()
        if num_frames is None or num_frames <= 0:
            num_frames = len([_ for _ in readers[0]])
            for r in readers: r.close()
            readers = [imageio.get_reader(p) for p in tile_paths]
            
        with imageio.get_writer(output_path, fps=fps, quality=quality) as writer:
            for start_frame in tqdm(range(0, num_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):
                end_frame = min(start_frame + chunk_size, num_frames)
                current_chunk_size = end_frame - start_frame
                chunk_canvas = np.zeros((current_chunk_size, final_H, final_W, 3), dtype=np.float32)
                weight_canvas = np.zeros_like(chunk_canvas, dtype=np.float32)
                
                for i, reader in enumerate(readers):
                    try:
                        tile_chunk_frames = [
                            frame.astype(np.float32) / 255.0 
                            for idx, frame in enumerate(reader.iter_data()) 
                            if start_frame <= idx < end_frame
                        ]
                        tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                    except Exception as e:
                        log(f"Warning: Could not read chunk from tile {i}. Error: {e}", message_type='warning')
                        continue
                    
                    if tile_chunk_np.shape[0] != current_chunk_size:
                        log(f"Warning: Tile {i} chunk has incorrect frame count. Skipping.", message_type='warning')
                        continue
                    
                    tile_H, tile_W, _ = tile_chunk_np.shape[1:]
                    ramp = np.linspace(0, 1, overlap * scale, dtype=np.float32)
                    mask = np.ones((tile_H, tile_W, 1), dtype=np.float32)
                    mask[:, :overlap*scale, :] *= ramp[np.newaxis, :, np.newaxis]
                    mask[:, -overlap*scale:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
                    mask[:overlap*scale, :, :] *= ramp[:, np.newaxis, np.newaxis]
                    mask[-overlap*scale:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
                    mask_4d = mask[np.newaxis, :, :, :]
                    
                    x1_orig, y1_orig, _, _ = tile_coords[i]
                    out_y1, out_x1 = y1_orig * scale, x1_orig * scale
                    out_y2, out_x2 = out_y1 + tile_H, out_x1 + tile_W
                    
                    chunk_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile_chunk_np * mask_4d
                    weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_4d
                    
                weight_canvas[weight_canvas == 0] = 1.0
                stitched_chunk = chunk_canvas / weight_canvas
                
                for frame_idx_in_chunk in range(current_chunk_size):
                    frame_uint8 = (np.clip(stitched_chunk[frame_idx_in_chunk], 0, 1) * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)
                    
    finally:
        log("Closing all tile reader instances...")
        for reader in readers:
            reader.close()
            
    if cleanup:
        log("Cleaning up temporary tile files...")
        for path in tile_paths:
            try:
                os.remove(path)
            except OSError as e:
                log(f"Could not remove temporary file '{path}': {e}", message_type='warning')

def init_pipeline(mode, device, dtype):
    model_downlod()
    model_path = os.path.join(ROOT_DIR, "models", "FlashVSR")
    ckpt_path, vae_path, lq_path, tcd_path, prompt_path = [os.path.join(model_path, f) for f in ["diffusion_pytorch_model_streaming_dmd.safetensors", "Wan2.1_VAE.pth", "LQ_proj_in.ckpt", "TCDecoder.ckpt", "../posi_prompt.pth"]]
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path]); pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
    else:
        mm.load_models([ckpt_path]); pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device=device, dtype=dtype, new_latent_channels=16+768)
        pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False); pipe.TCDecoder.clean_mem()
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    if os.path.exists(lq_path): pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.to(device, dtype=dtype); pipe.enable_vram_management(); pipe.init_cross_kv(prompt_path=prompt_path); pipe.load_models_to_device(["dit", "vae"])
    return pipe

# --- 集成化的核心逻辑函数 (更新版) ---
def run_flashvsr_integrated(
    input_path, 
    mode, 
    scale, 
    color_fix, 
    tiled_vae, 
    tiled_dit, 
    tile_size, 
    tile_overlap, 
    unload_dit, 
    dtype_str, 
    seed, 
    device, 
    fps_override,
    quality,
    attention_mode,
    sparse_ratio, # 新增
    kv_ratio,     # 新增
    local_range,  # 新增
    progress=gr.Progress(track_tqdm=True)
):
    if not input_path: raise gr.Error("请提供输入视频或图片文件夹路径！")
    if seed == -1: seed = random.randint(0, 2**32 - 1)
    
    # --- 参数准备 ---
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}; dtype = dtype_map.get(dtype_str, torch.bfloat16)
    devices = get_device_list(); _device = device
    if device == "auto": _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    if _device not in devices and _device != "cpu": raise gr.Error(f"设备 '{_device}' 不可用! 可用设备: {devices}")
    if _device.startswith("cuda"): torch.cuda.set_device(_device)
    if tiled_dit and (tile_overlap > tile_size / 2): raise gr.Error("重叠区域 (overlap) 必须小于分块大小 (tile_size) 的一半！")
    wan_video_dit.USE_BLOCK_ATTN = (attention_mode == "block")

    # --- 输出路径 ---
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"{input_basename}_{mode}_s{scale}_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # --- 核心逻辑 ---
    progress(0, desc="正在加载视频帧...")
    log(f"正在从 {input_path} 加载帧...", message_type='info')
    frames, original_fps = prepare_tensors(input_path, dtype=dtype)
    _fps = original_fps if is_video(input_path) else fps_override
    if frames.shape[0] < 21: raise gr.Error(f"输入帧数必须至少为21帧，当前为 {frames.shape[0]} 帧")
    log("视频帧加载完成。", message_type="finish")
    
    final_output_tensor = None
    
    # 构建通用 pipe 参数字典
    pipe_kwargs = {
        "prompt": "", "negative_prompt": "", "cfg_scale": 1.0, "num_inference_steps": 1,
        "seed": seed, "tiled": tiled_vae, "is_full_block": False, "if_buffer": True,
        "kv_ratio": kv_ratio, "local_range": local_range, "color_fix": color_fix,
        "unload_dit": unload_dit, "fps": _fps, "tiled_dit": tiled_dit,
    }

    if tiled_dit:
        N, H, W, C = frames.shape
        progress(0.1, desc="正在初始化模型管线...")
        pipe = init_pipeline(mode, _device, dtype)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        
        if mode == "tiny-long":
            local_temp_dir = os.path.join(TEMP_DIR, str(uuid.uuid4())); os.makedirs(local_temp_dir, exist_ok=True)
            temp_videos = []
            for i in tqdm(range(len(tile_coords)), desc="[FlashVSR] Processing tiles"):
                x1, y1, x2, y2 = tile_coords[i]
                input_tile = frames[:, y1:y2, x1:x2, :]
                temp_name = os.path.join(local_temp_dir, f"{i+1:05d}.mp4")
                th, tw, F = get_input_params(input_tile, scale)
                LQ_tile = input_tensor_generator(input_tile, _device, scale=scale, dtype=dtype)
                pipe(
                    LQ_video=LQ_tile, num_frames=F, height=th, width=tw, 
                    topk_ratio=sparse_ratio*768*1280/(th*tw),
                    quality=10, output_path=temp_name, **pipe_kwargs
                )
                temp_videos.append(temp_name); del LQ_tile, input_tile; clean_vram()
            
            stitch_video_tiles(temp_videos, tile_coords, (W*scale, H*scale), scale, tile_overlap, output_path, _fps, quality, True)
            shutil.rmtree(local_temp_dir)
        else: # 内存中拼接
            num_aligned_frames = largest_8n1_leq(N + 4) - 4
            final_output_canvas, weight_sum_canvas = torch.zeros((num_aligned_frames, H*scale, W*scale, C), dtype=torch.float32), torch.zeros((num_aligned_frames, H*scale, W*scale, C), dtype=torch.float32)
            for i in tqdm(range(len(tile_coords)), desc="[FlashVSR] Processing tiles"):
                x1, y1, x2, y2 = tile_coords[i]
                input_tile = frames[:, y1:y2, x1:x2, :]
                LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
                LQ_tile = LQ_tile.to(_device)
                output_tile_gpu = pipe(
                    LQ_video=LQ_tile, num_frames=F, height=th, width=tw,
                    topk_ratio=sparse_ratio*768*1280/(th*tw), **pipe_kwargs
                )
                processed_tile_cpu = tensor2video(output_tile_gpu).cpu()
                mask = create_feather_mask((processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]), tile_overlap * scale).cpu().permute(0, 2, 3, 1)
                x1_s, y1_s = x1 * scale, y1 * scale
                x2_s, y2_s = x1_s + processed_tile_cpu.shape[2], y1_s + processed_tile_cpu.shape[1]
                final_output_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += processed_tile_cpu * mask
                weight_sum_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += mask
                del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile; clean_vram()
            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            final_output_tensor = final_output_canvas / weight_sum_canvas
    else: # 非分块模式
        progress(0.1, desc="正在初始化模型管线...")
        pipe = init_pipeline(mode, _device, dtype)
        log(f"正在处理 {frames.shape[0]} 帧...", message_type='info')
        
        th, tw, F = get_input_params(frames, scale)
        if mode == "tiny-long":
            LQ = input_tensor_generator(frames, _device, scale=scale, dtype=dtype)
            pipe(
                LQ_video=LQ, num_frames=F, height=th, width=tw, 
                topk_ratio=sparse_ratio*768*1280/(th*tw),
                output_path=output_path, quality=quality, **pipe_kwargs
            )
        else:
            LQ, _, _, _ = prepare_input_tensor(frames, _device, scale=scale, dtype=dtype)
            LQ = LQ.to(_device)
            video = pipe(
                LQ_video=LQ, num_frames=F, height=th, width=tw, 
                topk_ratio=sparse_ratio*768*1280/(th*tw), **pipe_kwargs
            )
            final_output_tensor = tensor2video(video).cpu()
        del pipe; clean_vram()

    if final_output_tensor is not None:
        progress(0.9, desc="正在保存最终视频...")
        save_video(final_output_tensor, output_path, fps=_fps, quality=quality)
        
    log(f"处理完成！输出视频已保存至: {output_path}", message_type="finish")
    progress(1, desc="完成！")
    return output_path

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="FlashVSR+ WebUI") as demo:
        gr.Markdown("### FlashVSR+ WebUI")

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="上传视频文件", height=412)
                run_button = gr.Button("开始处理", variant="primary")
                gr.Markdown("### 主要设置")
                with gr.Group():
                    with gr.Row():
                        mode_radio = gr.Radio(choices=["tiny", "tiny-long", "full"], value="tiny", label="处理模式 (Pipeline Mode)")
                        seed_number = gr.Number(value=-1, label="随机种子 (Seed)", precision=0)
                with gr.Group():
                    with gr.Row():
                        scale_slider = gr.Slider(minimum=2, maximum=4, step=1, value=4, label="放大倍数 (Upscale Factor)")
                        tiled_dit_checkbox = gr.Checkbox(label="启用分块推理 (Tiled DiT)", info="适用于超大分辨率视频或显存不足的情况", value=False)
                    with gr.Row(visible=False) as tiled_dit_options:
                        tile_size_slider = gr.Slider(minimum=64, maximum=512, step=16, value=256, label="分块大小 (Tile Size)")
                        tile_overlap_slider = gr.Slider(minimum=8, maximum=128, step=8, value=24, label="重叠区域 (Tile Overlap)")

            with gr.Column(scale=1):
                video_output = gr.Video(label="输出结果", interactive=False, height=620)
                gr.Markdown("### 高级设置")
                with gr.Accordion("展开高级选项", open=False):
                    sparse_ratio_slider = gr.Slider(minimum=0.5, maximum=5.0, step=0.1, value=2.0, label="稀疏率 (Sparse Ratio)", info="控制注意力稀疏程度，值越小越稀疏")
                    kv_ratio_slider = gr.Slider(minimum=1, maximum=8, step=1, value=3, label="KV 缓存比率 (KV Ratio)", info="控制KV缓存长度")
                    local_range_slider = gr.Slider(minimum=3, maximum=15, step=2, value=11, label="局部范围 (Local Range)", info="局部注意力窗口大小")
                    attention_mode_radio = gr.Radio(choices=["sage", "block"], value="sage", label="注意力模式 (Attention Mode)")
                    color_fix_checkbox = gr.Checkbox(label="启用色彩校正 (Color Fix)", value=True)
                    tiled_vae_checkbox = gr.Checkbox(label="启用分块 VAE (Tiled VAE)", value=True)
                    unload_dit_checkbox = gr.Checkbox(label="解码前卸载 DiT (节省显存)", value=False)
                    dtype_radio = gr.Radio(choices=["fp16", "bf16"], value="bf16", label="数据类型 (Data Type)")
                    device_textbox = gr.Textbox(value="auto", label="运行设备 (Device)", info="例如: 'auto', 'cuda:0', 'cpu'")
                    quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=6, label="输出视频质量 (Quality)")
                    fps_number = gr.Number(value=30, label="输出帧率 (仅对图片序列输入有效)", precision=0)

        def toggle_tiled_dit_options(is_checked):
            return gr.update(visible=is_checked)
        
        tiled_dit_checkbox.change(fn=toggle_tiled_dit_options, inputs=[tiled_dit_checkbox], outputs=[tiled_dit_options])

        run_button.click(
            fn=run_flashvsr_integrated,
            inputs=[
                input_video, mode_radio, scale_slider, color_fix_checkbox, tiled_vae_checkbox, 
                tiled_dit_checkbox, tile_size_slider, tile_overlap_slider, unload_dit_checkbox, 
                dtype_radio, seed_number, device_textbox, fps_number, quality_slider, attention_mode_radio,
                sparse_ratio_slider, kv_ratio_slider, local_range_slider # 添加新参数
            ],
            outputs=[video_output]
        )

    return demo

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    model_downlod()
    ui = create_ui()
    if args.listen:
        ui.queue().launch(share=False, server_name="0.0.0.0", server_port=args.port)
    else:
        ui.queue().launch(share=False, server_port=args.port)