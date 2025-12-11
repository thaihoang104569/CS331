# LoRA DreamBooth Web App - Colab Setup

## 🚀 Quick Start on Google Colab

### Method 1: One-Click Colab Notebook

Open this notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thaihoang104569/gigi/blob/main/colab_inference.ipynb)

### Method 2: Manual Setup

```python
# 1. Clone repository
!git clone https://github.com/thaihoang104569/gigi.git
%cd gigi

# 2. Install dependencies
!pip install -q -r requirements.txt
!pip install -q gradio

# 3. Upload your LoRA weights (if not training on Colab)
from google.colab import files
uploaded = files.upload()  # Upload your lora_weight.safetensors

# 4. Launch web app
!python web_app.py
```

The app will create a public URL (like `https://xxxxx.gradio.live`) that you can access from any browser.

## 📱 Features

- **Simple Interface**: Load model, load LoRA, generate images
- **Full Control**: Adjust all parameters (alpha, steps, guidance scale, etc.)
- **Negative Prompts**: Improve quality by specifying what to avoid
- **Batch Comparison**: Generate multiple images with different alpha values
- **Seed Control**: Reproducible results

## 🎨 Usage

### 1. Setup Tab
1. **Load Base Model**: 
   - Default: `stablediffusionapi/realistic-vision-v51`
   - Optional: Add HF token to avoid rate limits
2. **Load LoRA Weights**:
   - Path: `output/lora_weight.safetensors`
   - Or custom path to your trained weights

### 2. Generate Tab
- **Prompt**: `professional photo of sks person, high quality, detailed`
- **Negative Prompt**: `ugly, blurry, low quality, distorted face`
- **LoRA Alpha**: `0.8` (recommended for portraits)
- **Steps**: `50` (higher = better quality)
- **Guidance Scale**: `7.5` (how closely to follow prompt)

### 3. Batch Compare Tab
- Compare different alpha values side-by-side
- Example: `0.0, 0.3, 0.5, 0.7, 1.0`

## 💡 Tips

### For Best Results:
- **Alpha 0.7-0.9**: Best for natural-looking portraits
- **Steps 40-60**: Good balance of quality and speed
- **Guidance 7-10**: Follows prompt without over-processing

### Common Issues:
- **Out of Memory**: Reduce resolution to 512x512
- **Slow Generation**: Lower steps to 30
- **Artifacts**: Reduce alpha below 1.0

## 📁 File Structure on Colab

```
/content/gigi/
├── web_app.py              # Web app
├── output/
│   ├── lora_weight.safetensors    # Your trained LoRA
│   └── loss_curve.png             # Training loss
├── lora_diffusion/         # LoRA library
└── requirements.txt
```

## 🔧 Advanced Configuration

### Custom Model:
```python
# In Setup tab, change Model ID to:
# - Lykon/dreamshaper-8
# - runwayml/stable-diffusion-v1-5
# - stabilityai/stable-diffusion-xl-base-1.0 (needs more VRAM)
```

### Multiple LoRAs:
Load different LoRAs to compare styles:
1. Load LoRA A → Generate
2. Restart runtime
3. Load LoRA B → Generate

## 🌐 Access from Phone/Tablet

The Gradio public URL works on any device:
1. Copy the `https://xxxxx.gradio.live` link
2. Open in mobile browser
3. Generate images on the go!

## ⚠️ Important Notes

- **Session Timeout**: Colab free tier has time limits
- **GPU Availability**: May need to wait for GPU allocation
- **Public URL**: Expires when notebook stops
- **File Persistence**: Download generated images before closing

## 🎓 Training on Colab

See main README for training instructions. Quick version:

```python
# Upload your photos
!mkdir -p my_photos
# Then upload via Colab UI

# Train
!python training_scripts/train_lora_dreambooth.py \
  --instance_data_dir my_photos \
  --instance_prompt "photo of sks person" \
  --output_dir output \
  --with_prior_preservation \
  --class_data_dir class_images \
  --class_prompt "photo of person" \
  --num_class_images 200 \
  --train_text_encoder \
  --max_train_steps 5000 \
  --lora_rank 16

# Launch web app
!python web_app.py
```

## 📞 Support

Issues? Check:
1. GPU is allocated: `!nvidia-smi`
2. Dependencies installed: `!pip list | grep diffusers`
3. LoRA file exists: `!ls -lh output/`

Happy generating! 🎨
