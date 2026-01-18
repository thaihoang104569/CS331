# LoRA DreamBooth Training

Hệ thống training và inference LoRA (Low-Rank Adaptation) cho Stable Diffusion với DreamBooth fine-tuning.

**Default Model**: Realistic Vision V5.1 (tốt nhất cho chân dung và khuôn mặt photorealistic)

## Giới thiệu LoRA

LoRA (Low-Rank Adaptation) là kỹ thuật fine-tuning hiệu quả:
- Thêm các ma trận nhỏ có thể train vào các layer hiện có
- Giữ nguyên base model (tiết kiệm bộ nhớ)
- Tạo file weights nhỏ gọn (1-6 MB thay vì 4-7 GB)
- Cho phép blend linh hoạt giữa base model và fine-tuned behavior

## Tính năng

✅ **DreamBooth Training**: Fine-tune Stable Diffusion trên ảnh của bạn  
✅ **Lightweight**: LoRA weights chỉ 1-6 MB  
✅ **Flexible**: Điều chỉnh mức độ ảnh hưởng từ 0.0 đến 1.0  
✅ **Textual Inversion**: Hỗ trợ custom tokens  
✅ **Demo Scripts**: So sánh trước/sau fine-tuning

## Cài đặt

```bash
# Clone repository
git clone https://github.com/thaihoang104569/CS331
cd CS331

# Cài đặt dependencies
pip install -r requirements.txt
```


## Yêu cầu

- Python 3.8+
- PyTorch 1.13+
- CUDA GPU với 12GB+ VRAM (khuyến nghị)

## Sử dụng

### 1. Chuẩn bị dữ liệu training

Tổ chức ảnh training:
```
my_training_data/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

Khuyến nghị: 5-20 ảnh của đối tượng/style của bạn.

### 2. Train LoRA Model

#### Training cơ bản (UNet only):
```bash
python training_scripts/train_lora_dreambooth.py \
  --pretrained_model_name_or_path="stablediffusionapi/realistic-vision-v51" \
  --instance_data_dir="my_training_data" \
  --output_dir="output/my_lora" \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --save_steps=1000 \
  --lora_rank=8 \
  --output_format="safe"
```

**Khuyến nghị cho face/person:**
- `--lora_rank=8`: Capacity cao hơn, học facial features tốt hơn
- `--max_train_steps=5000`: Đủ để converge với 10-20 ảnh
- `--gradient_accumulation_steps=4`: Smooth loss, ổn định hơn
- `--save_steps=1000`: Lưu checkpoint để so sánh

#### Training với Prior Preservation (Khuyên dùng - theo paper gốc):
```bash
python training_scripts/train_lora_dreambooth.py \
  --pretrained_model_name_or_path="stablediffusionapi/realistic-vision-v51" \
  --instance_data_dir="my_training_data" \
  --output_dir="output/my_lora_prior" \
  --instance_prompt="a photo of sks person" \
  --with_prior_preservation \
  --class_data_dir="class_images" \
  --class_prompt="a photo of person" \
  --num_class_images=200 \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --max_train_steps=5000 \
  --save_steps=1000 \
  --lora_rank=8 \
  --output_format="safe"
```

**Prior Preservation giúp:**
- ✅ **Tránh overfitting**: Model không quên kiến thức chung về class "person"
- ✅ **Tránh language drift**: Từ "person" vẫn giữ ý nghĩa gốc
- ✅ **Generalize tốt hơn**: Linh hoạt với pose, lighting, angle mới
- ✅ **Tự động generate**: Code sẽ tự tạo 200 ảnh class nếu chưa có

**Tham số quan trọng:**
- `--with_prior_preservation`: Bật prior preservation loss
- `--class_data_dir`: Thư mục chứa ảnh class (tự động tạo nếu chưa có)
- `--class_prompt`: Prompt cho class images (VD: "a photo of person")
- `--num_class_images`: Số ảnh class cần có (200 là tốt)
- `--prior_loss_weight`: Trọng số của prior loss (1.0 = cân bằng) (1.0 = cân bằng)

#### Training tối ưu nhất (Prior Preservation + Text Encoder):on + Text Encoder):
```bash
python training_scripts/train_lora_dreambooth.py \
  --pretrained_model_name_or_path="stablediffusionapi/realistic-vision-v51" \
  --instance_data_dir="my_training_data" \
  --output_dir="output/my_lora_best" \
  --instance_prompt="a photo of sks person" \
  --with_prior_preservation \
  --class_data_dir="class_images" \
  --class_prompt="a photo of person" \
  --num_class_images=200 \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
  --train_text_encoder \
  --color_jitter \
  --center_crop \
  --max_train_steps=6000 \
  --save_steps=1000 \
  --lora_rank=16 \
  --output_format="safe"
```

**Tối ưu cho facial features:**
- `--with_prior_preservation`: **Rất quan trọng** - tránh overfitting và language drift
- `--train_text_encoder`: **Quan trọng** - học được token = face identity
- `--lora_rank=16`: Capacity rất cao, chi tiết facial tốt nhất
- `--learning_rate_text=5e-5`: LR thấp hơn cho text encoder
- `--center_crop`: Focus vào center (mặt)
- `--num_class_images=200`: Đủ để giữ kiến thức class
- `--max_train_steps=6000`: Train lâu hơn với text encoder + prior preservation

**Sau khi train xong, bạn sẽ có:**
- `output/my_lora_best/lora_weight.safetensors` - LoRA weights
- `output/my_lora_best/loss_curve.png` - Biểu đồ loss theo training steps
- `output/my_lora_best/logs/` - TensorBoard logs để xem chi tiết

### 3. Test LoRA (Inference)

#### Demo script nhanh:
```bash
python demo_inference.py \
  --lora_path="output/my_lora/lora_weight.safetensors" \
  --prompt="a photo of sks person" \
  --alphas 0.0 0.5 1.0
```

Kết quả:
- `output/alpha_0.00.png` - Base model (trước fine-tuning)
- `output/alpha_0.50.png` - 50% LoRA
- `output/alpha_1.00.png` - Full LoRA (sau fine-tuning)
- `output/comparison_grid.png` - Grid so sánh

#### Code Python:
```python
import torch
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/realistic-vision-v51",
    torch_dtype=torch.float16
).to("cuda")

# Generate ảnh TRƯỚC fine-tuning
image_before = pipe("a photo of sks person", num_inference_steps=50).images[0]
image_before.save("before.png")

# Apply LoRA weights
patch_pipe(
    pipe,
    "output/my_lora/lora_weight.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True
)

# Set LoRA strength (0.0 = tắt, 1.0 = full)
tune_lora_scale(pipe.unet, 1.0)
tune_lora_scale(pipe.text_encoder, 1.0)

# Generate ảnh SAU fine-tuning
image_after = pipe("a photo of sks person", num_inference_steps=50).images[0]
image_after.save("after.png")
```

#### Web Interface (Gradio):
```bash
# Local
python web_app.py

# Google Colab
!git clone https://github.com/thaihoang104569/CS331.git
%cd CS331
!pip install -r requirements.txt
!python web_app.py
# Click vào public URL để dùng
```

Web interface cho phép:
- Load model và LoRA qua giao diện
- Nhập prompt và negative prompt
- Điều chỉnh alpha, steps, guidance scale
- So sánh nhiều alpha values
- Chạy trên Colab, truy cập từ bất kỳ thiết bị nào

**Chi tiết `web_app.py` (gợi ý workflow nhanh):**
- **Tab Setup**
  - **Model ID**: base model trên Hugging Face (mặc định: `stablediffusionapi/realistic-vision-v51`)
  - **HuggingFace Token** (tuỳ chọn): dùng khi model yêu cầu quyền truy cập
  - **LoRA Path**: đường dẫn tới file `.safetensors` (VD: `adapter/lora_weight.safetensors` hoặc `output/my_lora/lora_weight.safetensors`)
- **Tab Generate**
  - **LoRA Alpha**: 0.0 = base, 0.8 = khuyến nghị, 1.0 = full, >1.0 có thể tạo artifacts
  - **Steps / Guidance**: 30–50 steps và 7–10 guidance thường ổn cho portrait
  - **Seed**: đặt số cụ thể để tái lập ảnh; `-1` = random
  - **Width/Height**: mặc định 512; tăng lên 768/1024 sẽ tốn VRAM hơn
- **Tab Batch Compare**: nhập danh sách alpha dạng `0.0, 0.3, 0.5, 0.7, 1.0` để tìm mức alpha đẹp nhất

**Ghi chú (Local vs Colab):**
- Trong `web_app.py` mặc định `share=True` để tạo public URL (phù hợp Colab). Nếu chạy local và không cần public URL, có thể đổi `share=False`.
- Có thể set token bằng env var: `HF_TOKEN=...` (script có hỗ trợ đọc từ môi trường).

### 4. Đánh giá chất lượng bằng DINOv2 (cosine similarity)

File `evaluate_dino.py` tự động so sánh **base model** và **model + LoRA** bằng cách:
- Sinh ảnh từ prompt (base prompts vs finetuned prompts)
- Trích xuất embedding bằng **DINOv2 ViT-B/14**
- Tính cosine similarity giữa ảnh sinh và ảnh thật (ground truth)

Chạy nhanh:
```bash
python evaluate_dino.py \
  --real_images_dir "my_training_data" \
  --lora_path "output/my_lora/lora_weight.safetensors" \
  --lora_alpha 0.8 \
  --num_images_per_prompt 4 \
  --output_dir "evaluation_results/run_01"
```

Tuỳ chọn hay dùng:
- `--base_prompts`: prompt cho base model (không dùng rare token)
- `--finetuned_prompts`: prompt cho LoRA (có rare token, VD: `sks person`)
- `--num_inference_steps`, `--guidance_scale`, `--seed`: kiểm soát chất lượng và độ ổn định

Kết quả được lưu trong `--output_dir`:
- `base_model_images/` và `finetuned_model_images/`: ảnh sinh ra để soi trực quan
- `evaluation_results.json`: thống kê `mean/max/min similarity` + config
- `comparison_plot.png`: biểu đồ phân phối similarity và bảng so sánh

**Cách đọc metric:** mean similarity của finetuned cao hơn base (và tăng đủ lớn) thường là dấu hiệu LoRA học được identity/style.

**Lưu ý:** lần chạy đầu tiên sẽ tải model DINOv2 qua `torch.hub`, cần Internet và có thể hơi lâu.

## Tham số Training quan trọng

| Tham số | Mô tả | Khuyến nghị |
|---------|-------|-------------|
| `--lora_rank` | Rank của ma trận LoRA (cao hơn = capacity cao hơn) | UNet only: 8, With text: 16 |
| `--learning_rate` | Learning rate cho UNet | 1e-4 (ổn định) |
| `--learning_rate_text` | Learning rate cho text encoder | 5e-5 (thấp hơn UNet) |
| `--max_train_steps` | Tổng số bước training | UNet: 5000, With text: 6000 |
| `--gradient_accumulation_steps` | Accumulate gradients (smooth loss) | 4 (khuyến nghị cho stability) |
| `--train_text_encoder` | Fine-tune cả text encoder | **Bắt buộc cho face/person** |
| `--center_crop` | Crop center của ảnh | Khuyến nghị cho portrait |
| `--save_steps` | Lưu checkpoint mỗi N steps | 1000 (để so sánh) |
| `--gradient_checkpointing` | Giảm memory usage | Cho VRAM < 16GB |
| `--use_8bit_adam` | Dùng 8-bit Adam optimizer | Cho VRAM < 12GB |

**Tips quan trọng:**
- **Face/Person training:** Phải dùng `--train_text_encoder` + rank cao (16) để học facial features
- **Alpha inference:** Dùng 0.7-0.9 cho kết quả tốt nhất, tránh > 1.0 (gây distortion)
- **Prompt:** Dùng unique token (VD: `thaihoang_cs331`) thay vì generic `sks`
- **Convergence:** Loss nên < 0.05 sau 3000 steps, < 0.02 sau 5000 steps

## Tính năng nâng cao

### Điều chỉnh LoRA Strength
```python
# Test các mức độ ảnh hưởng khác nhau
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    tune_lora_scale(pipe.unet, alpha)
    tune_lora_scale(pipe.text_encoder, alpha)
    image = pipe(prompt).images[0]
    image.save(f"alpha_{alpha}.png")
```

### Merge nhiều LoRAs
```bash
lora_add \
  path/to/lora1.safetensors \
  path/to/lora2.safetensors \
  0.5 0.5 \
  output/merged_lora.safetensors
```

### Textual Inversion Support

LoRA files có thể bao gồm custom tokens (VD: `<s1><s2>`, `<sks>`):
```python
prompt = "a portrait of <s1><s2> in cyberpunk style"
```

Tokens được tự động load khi dùng `patch_pipe()`.

## Tối ưu bộ nhớ

Với GPU VRAM hạn chế:

```bash
python training_scripts/train_lora_dreambooth.py \
  --gradient_checkpointing \
  --use_8bit_adam \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  ...
```

Cho inference:
```python
# Dùng half precision
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")
```

## Cấu trúc Project

```
lora/
├── demo_inference.py              # Demo inference script
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── lora_diffusion/                # Core library
│   ├── lora.py                    # LoRA implementation
│   ├── utils.py                   # Utilities
│   ├── lora_manager.py            # Multi-LoRA management
│   └── xformers_utils.py          # Memory optimization
└── training_scripts/              # Training scripts
    └── train_lora_dreambooth.py   # Main DreamBooth trainer
```

## Xử lý lỗi

**Q: Kết quả inference không giống training data?**  
A: Kiểm tra:
- ✅ Prompt inference **phải khớp** với prompt training
- ✅ Alpha nên dùng 0.7-0.9, tránh > 1.0 gây distortion
- ✅ Phải dùng `--train_text_encoder` cho face/person training
- ✅ Tăng `--lora_rank` lên 16 nếu khuôn mặt không rõ

**Q: Khuôn mặt bị vỡ/distorted?**  
A: 
- Alpha quá cao (>1.0) → Giảm xuống 0.7-0.9
- Tăng `--guidance_scale=9.0` và `--num_inference_steps=80`
- Train lại với rank cao hơn (`--lora_rank=16`)

**Q: Model học được pose/clothes nhưng mặt sai?**  
A:
- **Bắt buộc** dùng `--train_text_encoder` + `--lora_rank=16`
- Thêm `--center_crop` để focus vào mặt
- Train lâu hơn: 6000 steps thay vì 3000

**Q: Out of memory khi training?**  
A: Dùng `--gradient_checkpointing`, `--use_8bit_adam`, giảm `--train_batch_size` xuống 1

**Q: Training quá chậm?**  
A: Cài `xformers`: `pip install xformers`, thêm flag `--use_xformers`

**Q: LoRA effect quá yếu/mạnh?**  
A: Điều chỉnh bằng `tune_lora_scale()` hoặc tăng/giảm `--lora_rank`

**Q: Ảnh không giống training data?**  
A: Train lâu hơn (`--max_train_steps`), dùng `--train_text_encoder`, hoặc tăng `--lora_rank`

## License

MIT License

## Author

thaihoang104569
