
import os
import torch
import gradio as gr
from pathlib import Path
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale
from PIL import Image
import numpy as np


class LoRAInferenceApp:
    def __init__(self):
        self.pipe = None
        self.lora_loaded = False
        self.current_lora_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_base_model(self, model_id, hf_token=None):
        """Load base Stable Diffusion model"""
        try:
            print(f"Loading base model: {model_id}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                resume_download=True,
                force_download=False,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                token=hf_token if hf_token else None,
            ).to(self.device)
            
            print(f"✓ Model loaded successfully on {self.device}")
            return f"✓ Model loaded: {model_id}"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def load_lora(self, lora_path):
        """Load LoRA weights"""
        if self.pipe is None:
            return "❌ Please load base model first!"
        
        try:
            if not os.path.exists(lora_path):
                return f"❌ LoRA file not found: {lora_path}"
            
            print(f"Loading LoRA: {lora_path}")
            patch_pipe(
                self.pipe,
                lora_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True,
            )
            self.lora_loaded = True
            self.current_lora_path = lora_path
            print("✓ LoRA loaded successfully")
            return f"✓ LoRA loaded: {os.path.basename(lora_path)}"
        except Exception as e:
            return f"❌ Error loading LoRA: {str(e)}"
    
    def generate_image(
        self,
        prompt,
        negative_prompt,
        lora_alpha,
        num_inference_steps,
        guidance_scale,
        seed,
        width,
        height
    ):
        """Generate image with current settings"""
        if self.pipe is None:
            return None, "❌ Please load base model first!"
        
        try:
            # Set LoRA alpha
            if self.lora_loaded:
                tune_lora_scale(self.pipe.unet, lora_alpha)
                tune_lora_scale(self.pipe.text_encoder, lora_alpha)
            
            # Set seed
            if seed == -1:
                seed = np.random.randint(0, 2**32 - 1)
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate
            print(f"Generating image with alpha={lora_alpha}, steps={num_inference_steps}")
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=width,
                    height=height,
                )
                image = result.images[0]
            
            info = f"✓ Generated successfully!\n"
            info += f"Seed: {seed}\n"
            info += f"LoRA Alpha: {lora_alpha}\n"
            info += f"Steps: {num_inference_steps}\n"
            info += f"Guidance Scale: {guidance_scale}"
            
            return image, info
            
        except Exception as e:
            return None, f"❌ Error generating image: {str(e)}"


# Initialize app
app = LoRAInferenceApp()


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="LoRA DreamBooth Inference", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 LoRA DreamBooth Inference
            Generate images using fine-tuned LoRA models
            """
        )
        
        with gr.Tab("Setup"):
            gr.Markdown("### 1. Load Base Model")
            with gr.Row():
                model_id = gr.Textbox(
                    value="stablediffusionapi/realistic-vision-v51",
                    label="Model ID",
                    placeholder="stablediffusionapi/realistic-vision-v51"
                )
                hf_token = gr.Textbox(
                    value="",
                    label="HuggingFace Token (optional)",
                    placeholder="hf_...",
                    type="password"
                )
            load_model_btn = gr.Button("Load Model", variant="primary")
            model_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### 2. Load LoRA Weights")
            with gr.Row():
                lora_path = gr.Textbox(
                    value="adapter/lora_weight.safetensors",
                    label="LoRA Path",
                    placeholder="adapter/lora_weight.safetensors"
                )
            load_lora_btn = gr.Button("Load LoRA", variant="primary")
            lora_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Prompt")
                    prompt = gr.Textbox(
                        value="professional photo of sks person, high quality, detailed face",
                        label="Prompt",
                        lines=3,
                        placeholder="Enter your prompt here..."
                    )
                    
                    negative_prompt = gr.Textbox(
                        value="ugly, blurry, low quality, distorted face, bad anatomy, bad hands, deformed",
                        label="Negative Prompt",
                        lines=2,
                        placeholder="What to avoid..."
                    )
                    
                    gr.Markdown("### Parameters")
                    
                    lora_alpha = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=0.8,
                        step=0.05,
                        label="LoRA Alpha (strength)",
                        info="0.0 = base model, 1.0 = full LoRA effect"
                    )
                    
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Inference Steps"
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    
                    with gr.Row():
                        width = gr.Dropdown(
                            choices=[512, 768, 1024],
                            value=512,
                            label="Width"
                        )
                        height = gr.Dropdown(
                            choices=[512, 768, 1024],
                            value=512,
                            label="Height"
                        )
                    
                    seed = gr.Number(
                        value=-1,
                        label="Seed (-1 for random)",
                        precision=0
                    )
                    
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_image = gr.Image(label="Generated Image", type="pil")
                    output_info = gr.Textbox(label="Generation Info", lines=5, interactive=False)
        
        with gr.Tab("Batch Compare"):
            gr.Markdown("### Compare different LoRA alphas")
            
            with gr.Row():
                batch_prompt = gr.Textbox(
                    value="professional photo of sks person",
                    label="Prompt",
                    lines=2
                )
            
            with gr.Row():
                alpha_values = gr.Textbox(
                    value="0.0, 0.3, 0.5, 0.7, 1.0",
                    label="Alpha Values (comma-separated)",
                    placeholder="0.0, 0.5, 1.0"
                )
                batch_steps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=5,
                    label="Steps per image"
                )
            
            batch_generate_btn = gr.Button("Generate Comparison", variant="primary")
            batch_gallery = gr.Gallery(label="Alpha Comparison", columns=5, height="auto")
            batch_info = gr.Textbox(label="Info", interactive=False)
        
        with gr.Tab("Help"):
            gr.Markdown(
                """
                ## 📖 How to Use
                
                ### Step 1: Setup
                1. **Load Base Model**: Enter model ID (default: Realistic Vision V5.1)
                2. **Load LoRA**: Enter path to your trained LoRA weights
                
                ### Step 2: Generate
                - **Prompt**: Describe what you want (use `sks person` for your trained subject)
                - **Negative Prompt**: What to avoid (helps improve quality)
                - **LoRA Alpha**: Control LoRA strength
                  - `0.0` = Original base model
                  - `0.8` = Recommended (balanced)
                  - `1.0` = Full LoRA effect
                  - `>1.0` = Stronger effect (may cause artifacts)
                - **Steps**: More = better quality but slower (30-50 is good)
                - **Guidance Scale**: How closely to follow prompt (7-10 recommended)
                
                ### Step 3: Batch Compare (Optional)
                - Generate multiple images with different alpha values
                - Great for finding optimal alpha
                
                ## 🚀 Tips
                - Start with alpha=0.8 for portraits
                - Use negative prompts to avoid common issues
                - Higher steps (50+) for final high-quality renders
                - Seed -1 for random, or set specific number for reproducibility
                
                ## 📁 File Paths on Colab
                - LoRA weights: `/content/CS331/output/lora_weight.safetensors`
                - Loss curve: `/content/CS331/output/loss_curve.png`
                """
            )
        
        # Event handlers
        load_model_btn.click(
            fn=lambda m, t: app.load_base_model(m, t if t else None),
            inputs=[model_id, hf_token],
            outputs=model_status
        )
        
        load_lora_btn.click(
            fn=app.load_lora,
            inputs=lora_path,
            outputs=lora_status
        )
        
        generate_btn.click(
            fn=app.generate_image,
            inputs=[
                prompt,
                negative_prompt,
                lora_alpha,
                num_inference_steps,
                guidance_scale,
                seed,
                width,
                height
            ],
            outputs=[output_image, output_info]
        )
        
        def batch_generate(prompt, alphas_str, steps):
            """Generate multiple images with different alphas"""
            if app.pipe is None:
                return [], "❌ Please load model first!"
            
            try:
                alphas = [float(a.strip()) for a in alphas_str.split(',')]
                images = []
                
                for alpha in alphas:
                    tune_lora_scale(app.pipe.unet, alpha)
                    tune_lora_scale(app.pipe.text_encoder, alpha)
                    
                    with torch.no_grad():
                        result = app.pipe(
                            prompt=prompt,
                            num_inference_steps=int(steps),
                            guidance_scale=7.5,
                        )
                        images.append(result.images[0])
                
                info = f"✓ Generated {len(images)} images with alphas: {alphas}"
                return images, info
                
            except Exception as e:
                return [], f"❌ Error: {str(e)}"
        
        batch_generate_btn.click(
            fn=batch_generate,
            inputs=[batch_prompt, alpha_values, batch_steps],
            outputs=[batch_gallery, batch_info]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    
    # Launch settings
    share = True  # Create public URL for Colab
    server_name = "0.0.0.0"  # Allow external connections
    server_port = 7860
    
    print("\n" + "="*60)
    print("🚀 Starting LoRA DreamBooth Web App...")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ Running on CPU (will be slow)")
    
    print("\n📝 Instructions:")
    print("1. Load base model (wait for confirmation)")
    print("2. Load your LoRA weights")
    print("3. Start generating!")
    print("="*60 + "\n")
    
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True
    )
