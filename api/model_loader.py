import torch
from diffusers import StableDiffusionXLPipeline

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_WEIGHTS_PATH = "training/checkpoints/pytorch_lora_weights.safetensors"

_pipeline = None


def get_pipeline():
    """
    Load the base SDXL pipeline with the fine-tuned LoRA weights applied,
    caching it as a module-level singleton so it loads only once per
    process, not once per request.
    """
    global _pipeline
    if _pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
        )
        pipeline.load_lora_weights(LORA_WEIGHTS_PATH)
        pipeline = pipeline.to(device)
        _pipeline = pipeline

    return _pipeline