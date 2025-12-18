from setuptools import setup, find_packages

setup(
    name="lora_diffusion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "diffusers>=0.11.0",
        "transformers>=4.25.1",
        "accelerate>=0.15.0",
        "safetensors>=0.3.0",
        "pillow>=9.0.0",
        "tqdm>=4.62.0",
        "scipy",
        "fire",
    ],
    python_requires=">=3.8",
)
