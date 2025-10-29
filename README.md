# ⚡ FlashVSR+

**Optimized inference pipeline based on [FlashVSR](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) project**

**Authors:** Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, Tianfan Xue

**Modified:** Naxci1  

<a href='http://zhuang2002.github.io/FlashVSR'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/JunhaoZhuang/FlashVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a> &nbsp;
<a href="https://huggingface.co/datasets/JunhaoZhuang/VSR-120K"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a> &nbsp;
<a href="https://arxiv.org/abs/2510.12747"><img src="https://img.shields.io/badge/arXiv-2510.12747-b31b1b.svg"></a>

**Your star means a lot for us to develop this project!** :star:

<img src="./teaser.jpg" />

---
### 🤔 What's New?

- Replaced `Block-Sparse-Attention` with `Sparse_SageAttention` to avoid building complex cuda kernels.  
- With the new `tile_dit` method, you can even output 1080P video on 8GB of VRAM.   
- Support copying audio tracks to output files (powered by FFmpeg). 
- Introduced Blackwell GPU support for FlashVSR.  

---
### 🚀 Getting Started

Follow these steps to set up and run **FlashVSR** on your local machine:

> ⚠️ **Note:** This project is primarily designed and optimized for **4× video super-resolution**.  
> We **strongly recommend** using the **4× SR setting** to achieve better results and stability. ✅



#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/lihaoyun6/FlashVSR_Stable
cd FlashVSR_Stable
````

#### 2️⃣ Set Up the Python Environment

Create and activate the environment:

```bash
conda create -n flashvsr
conda activate flashvsr
```

Install project dependencies:

```bash
# for CUDA 12.8
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128

# for CUDA 13.0
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
```

#### 3️⃣ Download Model Weights

- When you run FlashVSR+ for the first time, it will automatically download all required models from HuggingFace.  

- You can also manually download all files from [FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR) and put them in the following location:  

```
./models/FlashVSR/
│
├── LQ_proj_in.ckpt                                   
├── TCDecoder.ckpt                                    
├── Wan2.1_VAE.pth                                    
├── diffusion_pytorch_model_streaming_dmd.safetensors 
└── README.md
```  

#### 4️⃣ Run Inference

CLI example:

```bash

python run.py -i "C:\video\input" -o "C:\video\output" -s 2 -m full --tiled-vae --tiled-dit --tile-size 368 --overlap 24 --color-fix -t bf16  --batch-size 200

python run.py -i "C:\video\input\split" -o "C:\video\output" -s 4 -m full --tiled-vae --tiled-dit --tile-size 256 --overlap 24 --color-fix -t bf16 --max-frames 100 --batch-size 100
```


---

### 🤗 Feedback & Support

We welcome feedback and issues. Thank you for trying **FlashVSR_Stable**

---

### 📄 Acknowledgments

We gratefully acknowledge the following open-source projects:

* **FlashVSR** — [https://github.com/OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
* **DiffSynth Studio** — [https://github.com/modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* **Sparse_SageAttention** — [https://github.com/jt-zhang/Sparse\_SageAttention_API](https://github.com/jt-zhang/Sparse_SageAttention_API)
* **taehv** — [https://github.com/madebyollin/taehv](https://github.com/madebyollin/taehv)

---


