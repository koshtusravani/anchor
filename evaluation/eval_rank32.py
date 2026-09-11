import json
import torch
from diffusers import StableDiffusionXLPipeline
from eval_suite import (
    PROMPT_SUITE, INSTANCE_DATA_DIR, OUTPUT_DIR, DEVICE,
    load_clip_model, compute_clip_i, compute_clip_t, generate_images
)
import os

LORA_RANK32_PATH = "/root/anchor/checkpoints/lora_colorful_sneaker_rank32/pytorch_lora_weights.safetensors"

def main():
    reference_images = [
        os.path.join(INSTANCE_DATA_DIR, f)
        for f in os.listdir(INSTANCE_DATA_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    clip_model, clip_preprocess, clip_tokenizer = load_clip_model()

    print("Generating with rank-32 fine-tuned LoRA model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(DEVICE)
    pipe.load_lora_weights(LORA_RANK32_PATH)

    images = generate_images(pipe, PROMPT_SUITE, f"{OUTPUT_DIR}/finetuned_rank32")

    results = []
    for prompt, image in zip(PROMPT_SUITE, images):
        clip_i = compute_clip_i(clip_model, clip_preprocess, image, reference_images)
        clip_t = compute_clip_t(clip_model, clip_preprocess, clip_tokenizer, image, prompt)
        results.append({"prompt": prompt, "clip_i": clip_i, "clip_t": clip_t})

    avg_i = sum(r["clip_i"] for r in results) / len(results)
    avg_t = sum(r["clip_t"] for r in results) / len(results)
    print(f"rank32: avg CLIP-I={avg_i:.4f}, avg CLIP-T={avg_t:.4f}")

    with open("/root/anchor/evaluation/results_rank32.json", "w") as f:
        json.dump({"rank32": results, "rank32_summary": {"avg_clip_i": avg_i, "avg_clip_t": avg_t}}, f, indent=2)

if __name__ == "__main__":
    main()
