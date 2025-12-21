# Đánh giá Model bằng DINO Metric

Script này đánh giá model Stable Diffusion sau khi fine-tune bằng LoRA, sử dụng DINO metric để so sánh identity preservation.

## Mục đích

- **Chứng minh base model không có identity của bạn**: Base model chưa fine-tune sẽ có cosine similarity thấp với ảnh thật của bạn
- **Chứng minh finetuned model đã học được identity**: Model sau fine-tune sẽ có cosine similarity cao hơn đáng kể
- **So sánh định lượng**: Sử dụng DINO embeddings và cosine similarity để đo lường chính xác

## Cài đặt thư viện cần thiết

```bash
# Cài đặt các thư viện cơ bản (nếu chưa có)
pip install torch torchvision diffusers transformers accelerate safetensors pillow tqdm matplotlib seaborn numpy

# DINO sẽ được tải tự động từ torch.hub khi chạy script
```

## Cách sử dụng

### 1. Script đầy đủ (đánh giá chi tiết)

```bash
python evaluate_dino.py \
    --real_images_dir data/real_images \
    --lora_path adapter/lora_weight.safetensors \
    --prompts "a photo of sks person" "a portrait of sks person" "sks person smiling" \
    --num_images_per_prompt 5 \
    --output_dir evaluation_results
```

**Output:**
- `evaluation_results/base_model_images/`: Ảnh sinh từ base model
- `evaluation_results/finetuned_model_images/`: Ảnh sinh từ finetuned model
- `evaluation_results/evaluation_results.json`: Kết quả chi tiết dưới dạng JSON
- `evaluation_results/comparison_plot.png`: Biểu đồ so sánh trực quan

### 2. Script nhanh (đánh giá nhanh chóng)

Để test nhanh hơn với ít ảnh hơn:

```bash
python evaluate_dino_quick.py \
    --real_images_dir data/real_images \
    --lora_path adapter/lora_weight.safetensors \
    --prompts "a photo of sks person" \
    --num_images 3 \
    --output_dir evaluation_quick
```

## Tham số quan trọng

### Tham số bắt buộc:
- `--real_images_dir`: Thư mục chứa ảnh thật của bạn (ground truth)
- `--lora_path`: Đường dẫn đến file LoRA weights (`.pt` hoặc `.safetensors`)

### Tham số tùy chọn:
- `--model_id`: Base model (mặc định: `stablediffusionapi/realistic-vision-v51`)
- `--prompts`: Danh sách prompts để sinh ảnh (sử dụng token `sks person`)
- `--num_images_per_prompt`: Số ảnh sinh cho mỗi prompt (mặc định: 5)
- `--num_inference_steps`: Số bước denoising (mặc định: 50)
- `--guidance_scale`: CFG scale (mặc định: 7.5)
- `--seed`: Random seed (mặc định: 42)
- `--device`: `cuda` hoặc `cpu`
- `--hf_token`: Hugging Face token (hoặc đặt biến môi trường `HF_TOKEN`)

## Hiểu kết quả

### Cosine Similarity
- **0.0 - 0.3**: Rất khác biệt (không có identity)
- **0.3 - 0.5**: Khác biệt vừa phải
- **0.5 - 0.7**: Tương đồng khá cao
- **0.7 - 1.0**: Rất tương đồng (identity được bảo toàn tốt)

### Ví dụ kết quả mong đợi

```
--- BASE MODEL RESULTS ---
Mean Similarity: 0.3245 ± 0.0823
Max Similarity:  0.4512
Min Similarity:  0.1893
Median:          0.3189

--- FINETUNED MODEL RESULTS ---
Mean Similarity: 0.6834 ± 0.0512
Max Similarity:  0.7923
Min Similarity:  0.5621
Median:          0.6891

--- COMPARISON ---
Improvement: +0.3589 (+110.64%)

✓ FINETUNED MODEL SUCCESSFULLY LEARNED YOUR IDENTITY!
✓ BASE MODEL DOES NOT HAVE YOUR IDENTITY
```

## DINO Metric là gì?

**DINO (Self-Distillation with No Labels)** là một vision transformer model được train bằng self-supervised learning. DINOv2 đặc biệt hiệu quả cho:

- **Identity preservation**: Đo lường xem model có giữ được đặc điểm nhận dạng không
- **Semantic similarity**: So sánh tương đồng về mặt ngữ nghĩa giữa các ảnh
- **Robust features**: Embeddings robust với lighting, pose, expression variations

### Tại sao dùng DINO thay vì metrics khác?

- **CLIP**: Tốt cho text-image alignment nhưng kém cho identity
- **Face Recognition (ArcFace, FaceNet)**: Chỉ work với ảnh mặt, không general
- **DINO**: Tốt nhất cho identity + general purpose, không cần bounding box

## Ví dụ sử dụng

### Ví dụ 1: Đánh giá với data của bạn

```bash
# Chuẩn bị data
# - Đặt ảnh thật vào: data/my_photos/
# - Training LoRA đã hoàn thành: output/lora_weights.safetensors

# Chạy đánh giá
python evaluate_dino.py \
    --real_images_dir data/my_photos \
    --lora_path output/lora_weights.safetensors \
    --prompts "photo of sks person" "portrait of sks person" "sks person in a suit" \
    --num_images_per_prompt 5 \
    --output_dir results/dino_evaluation
```

### Ví dụ 2: Test nhanh

```bash
python evaluate_dino_quick.py \
    --real_images_dir data/my_photos \
    --lora_path output/lora_weights.safetensors \
    --num_images 2 \
    --output_dir results/quick_test
```

### Ví dụ 3: So sánh nhiều LoRA

```bash
# Đánh giá LoRA version 1
python evaluate_dino.py \
    --real_images_dir data/my_photos \
    --lora_path experiments/lora_v1.safetensors \
    --output_dir results/lora_v1

# Đánh giá LoRA version 2
python evaluate_dino.py \
    --real_images_dir data/my_photos \
    --lora_path experiments/lora_v2.safetensors \
    --output_dir results/lora_v2

# So sánh results/lora_v1/evaluation_results.json và results/lora_v2/evaluation_results.json
```

## Troubleshooting

### Lỗi: "No images found in directory"
- Kiểm tra `--real_images_dir` có đúng đường dẫn không
- Đảm bảo có ảnh với định dạng: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### Lỗi CUDA out of memory
- Giảm `--num_images_per_prompt`
- Sử dụng `--device cpu` (chậm hơn)
- Đóng các ứng dụng khác đang dùng GPU

### Lỗi Hugging Face rate limit
- Đăng ký tài khoản Hugging Face: https://huggingface.co/
- Lấy token: https://huggingface.co/settings/tokens
- Đặt token: `export HF_TOKEN=your_token_here` hoặc `--hf_token your_token_here`

### DINO download chậm
- DINO model (~300MB) sẽ được download lần đầu tiên
- Lần sau sẽ load từ cache: `~/.cache/torch/hub/`

## Kết quả mong đợi

Một lần đánh giá thành công sẽ cho thấy:

1. **Base model có similarity thấp (< 0.4)**: Chứng minh base model không biết identity của bạn
2. **Finetuned model có similarity cao (> 0.6)**: Chứng minh model đã học được identity
3. **Improvement > 50%**: Model fine-tune tốt
4. **Visual comparison**: Ảnh finetuned trông giống bạn hơn base model

## Trích dẫn (Citation)

Nếu sử dụng trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Theo and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## Liên hệ

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue hoặc liên hệ.
