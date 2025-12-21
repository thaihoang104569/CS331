"""
Script đánh giá model bằng DINO metric để so sánh base model và finetuned model.
Trích xuất embedding và tính cosine similarity giữa ảnh sinh và ảnh thật.

Usage:
    python evaluate_dino.py --real_images_dir data/real_images --lora_path adapter/lora_weight.safetensors
"""

import argparse
import os
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns


class DINOEvaluator:
    """Đánh giá model sử dụng DINOv2 embeddings"""
    
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Loading DINOv2 model...")
        
        # Load DINOv2 ViT-B/14 model
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dino_model = self.dino_model.to(device).eval()
        
        # Transform cho DINO (chuẩn hóa theo ImageNet)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        print("✓ DINOv2 model loaded successfully")
    
    @torch.no_grad()
    def extract_embedding(self, image):
        """Trích xuất DINO embedding từ ảnh PIL"""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        # Transform và move to device
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        embedding = self.dino_model(img_tensor)
        
        # Normalize embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()[0]
    
    def cosine_similarity(self, emb1, emb2):
        """Tính cosine similarity giữa 2 embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def evaluate_images(self, real_images, generated_images):
        """
        Đánh giá tập ảnh sinh ra so với ảnh thật
        
        Returns:
            dict: Kết quả đánh giá với các metrics
        """
        real_embeddings = []
        gen_embeddings = []
        
        print("Extracting embeddings from real images...")
        for img in tqdm(real_images):
            emb = self.extract_embedding(img)
            real_embeddings.append(emb)
        
        print("Extracting embeddings from generated images...")
        for img in tqdm(generated_images):
            emb = self.extract_embedding(img)
            gen_embeddings.append(emb)
        
        # Tính cosine similarity giữa từng cặp
        similarities = []
        for real_emb in real_embeddings:
            for gen_emb in gen_embeddings:
                sim = self.cosine_similarity(real_emb, gen_emb)
                similarities.append(sim)
        
        # Tính các metrics
        results = {
            'mean_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'similarities': [float(s) for s in similarities],
            'num_real_images': len(real_images),
            'num_generated_images': len(generated_images),
        }
        
        return results


def load_images_from_directory(directory):
    """Load tất cả ảnh từ thư mục"""
    image_paths = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    directory = Path(directory)
    for ext in valid_extensions:
        image_paths.extend(list(directory.glob(f'*{ext}')))
        image_paths.extend(list(directory.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_paths)} images in {directory}")
    return sorted(image_paths)


def generate_images(pipe, prompts, num_images_per_prompt=5, seed=42, 
                   num_inference_steps=50, guidance_scale=7.5, negative_prompt=None):
    """Sinh ảnh từ Stable Diffusion pipeline"""
    images = []
    
    for prompt in tqdm(prompts, desc="Generating images"):
        for i in range(num_images_per_prompt):
            # Set seed khác nhau cho mỗi ảnh
            torch.manual_seed(seed + i)
            
            with torch.no_grad():
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            
            images.append(image)
    
    return images


def plot_comparison(results_base, results_finetuned, output_path):
    """Vẽ biểu đồ so sánh kết quả"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar chart - Mean similarity
    ax = axes[0, 0]
    models = ['Base Model', 'Finetuned Model']
    means = [results_base['mean_similarity'], results_finetuned['mean_similarity']]
    
    bars = ax.bar(models, means, alpha=0.7, color=['#ff6b6b', '#4ecdc4'])
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Mean DINO Cosine Similarity (Higher = Better Identity Preservation)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị lên cột
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Distribution comparison
    ax = axes[0, 1]
    ax.hist(results_base['similarities'], bins=30, alpha=0.6, label='Base Model', color='#ff6b6b', density=False)
    ax.hist(results_finetuned['similarities'], bins=30, alpha=0.6, label='Finetuned Model', color='#4ecdc4', density=False)
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Cosine Similarities', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # 3. Box plot
    ax = axes[1, 0]
    data_to_plot = [results_base['similarities'], results_finetuned['similarities']]
    bp = ax.boxplot(data_to_plot, labels=models, patch_artist=True)
    
    colors = ['#ff6b6b', '#4ecdc4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Base Model', 'Finetuned Model', 'Improvement'],
        ['Mean', f"{results_base['mean_similarity']:.4f}", f"{results_finetuned['mean_similarity']:.4f}", 
         f"{(results_finetuned['mean_similarity'] - results_base['mean_similarity']):.4f}"],
        ['Max', f"{results_base['max_similarity']:.4f}", f"{results_finetuned['max_similarity']:.4f}", 
         f"{(results_finetuned['max_similarity'] - results_base['max_similarity']):.4f}"],
        ['Min', f"{results_base['min_similarity']:.4f}", f"{results_finetuned['min_similarity']:.4f}", 
         f"{(results_finetuned['min_similarity'] - results_base['min_similarity']):.4f}"],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color improvement column
    for i in range(1, len(table_data)):
        improvement = float(table_data[i][3])
        if improvement > 0:
            table[(i, 3)].set_facecolor('#d4edda')  # Green for positive
        elif improvement < 0:
            table[(i, 3)].set_facecolor('#f8d7da')  # Red for negative
    
    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Đánh giá model bằng DINO metric")
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="stablediffusionapi/realistic-vision-v51",
        help="Pretrained Stable Diffusion model ID"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA weights (.pt or .safetensors)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="LoRA alpha/scale (0.0=no LoRA, 1.0=full LoRA, >1.0=over-apply). Default: 1.0"
    )
    
    # Data arguments
    parser.add_argument(
        "--real_images_dir",
        type=str,
        required=True,
        help="Directory containing real images (ground truth)"
    )
    parser.add_argument(
        "--base_prompts",
        type=str,
        nargs="+",
        default=["a photo of a person", "a portrait of a man", "a person smiling"],
        help="Prompts for base model (without rare token)"
    )
    parser.add_argument(
        "--finetuned_prompts",
        type=str,
        nargs="+",
        default=["a photo of sks person", "a portrait of sks person", "sks person smiling"],
        help="Prompts for finetuned model (with rare token like 'sks person')"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=5,
        help="Number of images to generate per prompt"
    )
    
    # Generation arguments
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="ugly, blurry, low quality, distorted face, bad anatomy, bad hands, deformed",
        help="Negative prompt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token"
    )
    
    args = parser.parse_args()
    
    # Setup HF token
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN", None)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Subdirectories for generated images
    base_images_dir = output_dir / "base_model_images"
    finetuned_images_dir = output_dir / "finetuned_model_images"
    base_images_dir.mkdir(exist_ok=True)
    finetuned_images_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("DINO METRIC EVALUATION - BASE MODEL vs FINETUNED MODEL")
    print("="*70)
    
    # Load real images
    print(f"\n[1/6] Loading real images from {args.real_images_dir}")
    real_image_paths = load_images_from_directory(args.real_images_dir)
    if len(real_image_paths) == 0:
        raise ValueError(f"No images found in {args.real_images_dir}")
    
    # Initialize DINO evaluator
    print(f"\n[2/6] Initializing DINO evaluator")
    evaluator = DINOEvaluator(device=args.device)
    
    # ============ EVALUATE BASE MODEL ============
    print("\n" + "="*70)
    print("EVALUATING BASE MODEL (without LoRA)")
    print("="*70)
    
    print(f"\n[3/6] Loading base model: {args.model_id}")
    pipe_base = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True,
        token=args.hf_token,
    ).to(args.device)
    
    print(f"\n[4/6] Generating images from base model...")
    print(f"Using generic prompts (no rare token): {args.base_prompts}")
    base_images = generate_images(
        pipe_base,
        args.base_prompts,
        num_images_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
    )
    
    # Save base model images
    for i, img in enumerate(base_images):
        img.save(base_images_dir / f"image_{i:03d}.png")
    print(f"✓ Saved {len(base_images)} base model images to {base_images_dir}")
    
    # Evaluate base model
    print(f"\n[5/6] Evaluating base model with DINO metric...")
    results_base = evaluator.evaluate_images(real_image_paths, base_images)
    
    print("\n--- BASE MODEL RESULTS ---")
    print(f"Mean Similarity: {results_base['mean_similarity']:.4f}")
    print(f"Max Similarity:  {results_base['max_similarity']:.4f}")
    print(f"Min Similarity:  {results_base['min_similarity']:.4f}")
    
    # Clean up base model to free memory
    del pipe_base
    torch.cuda.empty_cache()
    
    # ============ EVALUATE FINETUNED MODEL ============
    print("\n" + "="*70)
    print("EVALUATING FINETUNED MODEL (with LoRA)")
    print("="*70)
    
    print(f"\n[6/6] Loading finetuned model with LoRA: {args.lora_path}")
    pipe_finetuned = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True,
        token=args.hf_token,
    ).to(args.device)
    
    # Apply LoRA
    patch_pipe(
        pipe_finetuned,
        args.lora_path,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )
    
    # Set LoRA alpha/scale
    tune_lora_scale(pipe_finetuned.unet, args.lora_alpha)
    tune_lora_scale(pipe_finetuned.text_encoder, args.lora_alpha)
    print(f"LoRA alpha set to: {args.lora_alpha}")
    
    print(f"\nGenerating images from finetuned model...")
    print(f"Using prompts with rare token: {args.finetuned_prompts}")
    finetuned_images = generate_images(
        pipe_finetuned,
        args.finetuned_prompts,
        num_images_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
    )
    
    # Save finetuned model images
    for i, img in enumerate(finetuned_images):
        img.save(finetuned_images_dir / f"image_{i:03d}.png")
    print(f"✓ Saved {len(finetuned_images)} finetuned model images to {finetuned_images_dir}")
    
    # Evaluate finetuned model
    print(f"\nEvaluating finetuned model with DINO metric...")
    results_finetuned = evaluator.evaluate_images(real_image_paths, finetuned_images)
    
    print("\n--- FINETUNED MODEL RESULTS ---")
    print(f"Mean Similarity: {results_finetuned['mean_similarity']:.4f}")
    print(f"Max Similarity:  {results_finetuned['max_similarity']:.4f}")
    print(f"Min Similarity:  {results_finetuned['min_similarity']:.4f}")
    
    # ============ COMPARISON ============
    print("\n" + "="*70)
    print("COMPARISON & CONCLUSION")
    print("="*70)
    
    improvement = results_finetuned['mean_similarity'] - results_base['mean_similarity']
    improvement_pct = (improvement / results_base['mean_similarity']) * 100
    
    print(f"\nImprovement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print(f"Base Model Mean:      {results_base['mean_similarity']:.4f}")
    print(f"Finetuned Model Mean: {results_finetuned['mean_similarity']:.4f}")
    
    # Conclusion
    print("\n--- CONCLUSION ---")
    if improvement > 0.05:  # Significant improvement threshold
        print("✓ FINETUNED MODEL SUCCESSFULLY LEARNED YOUR IDENTITY!")
        print(f"  The finetuned model shows {improvement_pct:.1f}% higher similarity to your real images,")
        print(f"  indicating that it has successfully captured your identity.")
    elif improvement > 0.01:
        print("⚠ FINETUNED MODEL SHOWS MODERATE IMPROVEMENT")
        print(f"  The finetuned model shows {improvement_pct:.1f}% improvement, but it's relatively modest.")
        print(f"  Consider training for more steps or adjusting hyperparameters.")
    else:
        print("✗ FINETUNED MODEL DID NOT SIGNIFICANTLY IMPROVE")
        print(f"  The improvement is only {improvement_pct:.1f}%, which is not significant.")
        print(f"  The model may need more training data or different hyperparameters.")
    
    if results_base['mean_similarity'] < 0.4:
        print(f"\n✓ BASE MODEL DOES NOT HAVE YOUR IDENTITY")
        print(f"  Low similarity ({results_base['mean_similarity']:.4f}) confirms the base model")
        print(f"  does not know your identity before fine-tuning.")
    
    # Save results to JSON
    all_results = {
        'base_model': results_base,
        'finetuned_model': results_finetuned,
        'improvement': float(improvement),
        'improvement_percentage': float(improvement_pct),
        'config': {
            'model_id': args.model_id,
            'lora_path': args.lora_path,
            'lora_alpha': args.lora_alpha,
            'base_prompts': args.base_prompts,
            'finetuned_prompts': args.finetuned_prompts,
            'num_images_per_prompt': args.num_images_per_prompt,
            'seed': args.seed,
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': args.guidance_scale,
        }
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Plot comparison
    print(f"\nCreating comparison plots...")
    plot_path = output_dir / "comparison_plot.png"
    plot_comparison(results_base, results_finetuned, plot_path)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}")
    print(f"  - Base model images: {base_images_dir}")
    print(f"  - Finetuned model images: {finetuned_images_dir}")
    print(f"  - Evaluation results: {results_path}")
    print(f"  - Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
