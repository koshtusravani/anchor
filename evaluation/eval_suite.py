import os
import json
import torch
import open_clip
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel


# Fixed prompt suite spanning pose, setting, and style variation, used
# identically across all three comparison arms.
PROMPT_SUITE = [
    "a photo of sks sneaker on a wooden table",
    "a photo of sks sneaker in the snow",
    "a photo of sks sneaker on top of a mountain",
    "a photo of sks sneaker floating on water",
    "a photo of sks sneaker next to a coffee cup",
    "a painting of sks sneaker in the style of Van Gogh",
    "a photo of sks sneaker on a red carpet",
    "a photo of sks sneaker in a forest",
    "a close-up photo of sks sneaker",
    "a photo of sks sneaker on the moon",
]

INSTANCE_DATA_DIR = "/root/data/raw/dataset/colorful_sneaker"
LORA_WEIGHTS_PATH = "/root/anchor/training/checkpoints/pytorch_lora_weights.safetensors"
OUTPUT_DIR = "/root/anchor/evaluation/generated"
RESULTS_PATH = "/root/anchor/evaluation/results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model():
    """Load a CLIP model for computing CLIP-I and CLIP-T scores."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(DEVICE).eval()
    return model, preprocess, tokenizer


def compute_clip_i(model, preprocess, generated_image, reference_images):
    """
    Compute CLIP-I: average cosine similarity between the generated image
    and each reference image, measuring subject fidelity.
    """
    with torch.no_grad():
        gen_tensor = preprocess(generated_image).unsqueeze(0).to(DEVICE)
        gen_features = model.encode_image(gen_tensor)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

        similarities = []
        for ref_path in reference_images:
            ref_image = Image.open(ref_path).convert("RGB")
            ref_tensor = preprocess(ref_image).unsqueeze(0).to(DEVICE)
            ref_features = model.encode_image(ref_tensor)
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
            sim = (gen_features @ ref_features.T).item()
            similarities.append(sim)

    return sum(similarities) / len(similarities)


def compute_clip_t(model, preprocess, tokenizer, generated_image, prompt):
    """
    Compute CLIP-T: cosine similarity between the generated image and the
    text prompt, measuring prompt fidelity.
    """
    with torch.no_grad():
        image_tensor = preprocess(generated_image).unsqueeze(0).to(DEVICE)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_tokens = tokenizer([prompt]).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        sim = (image_features @ text_features.T).item()

    return sim


def generate_images(pipe, prompts, output_subdir):
    """Generate one image per prompt and save to output_subdir."""
    os.makedirs(output_subdir, exist_ok=True)
    images = []
    for i, prompt in enumerate(prompts):
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        path = os.path.join(output_subdir, f"{i:02d}.png")
        image.save(path)
        images.append(image)
    return images


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reference_images = [
        os.path.join(INSTANCE_DATA_DIR, f)
        for f in os.listdir(INSTANCE_DATA_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    clip_model, clip_preprocess, clip_tokenizer = load_clip_model()

    results = {"base": [], "finetuned": []}

    # Arm 1: frozen base SDXL, text-only prompting
    print("Generating with base SDXL...")
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(DEVICE)

    base_prompts = [p.replace("sks ", "") for p in PROMPT_SUITE]
    base_images = generate_images(base_pipe, base_prompts, f"{OUTPUT_DIR}/base")

    for prompt, image in zip(base_prompts, base_images):
        clip_i = compute_clip_i(clip_model, clip_preprocess, image, reference_images)
        clip_t = compute_clip_t(clip_model, clip_preprocess, clip_tokenizer, image, prompt)
        results["base"].append({"prompt": prompt, "clip_i": clip_i, "clip_t": clip_t})

    del base_pipe
    torch.cuda.empty_cache()

    # Arm 2: fine-tuned LoRA model
    print("Generating with fine-tuned LoRA model...")
    ft_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(DEVICE)
    ft_pipe.load_lora_weights(LORA_WEIGHTS_PATH)

    ft_images = generate_images(ft_pipe, PROMPT_SUITE, f"{OUTPUT_DIR}/finetuned")

    for prompt, image in zip(PROMPT_SUITE, ft_images):
        clip_i = compute_clip_i(clip_model, clip_preprocess, image, reference_images)
        clip_t = compute_clip_t(clip_model, clip_preprocess, clip_tokenizer, image, prompt)
        results["finetuned"].append({"prompt": prompt, "clip_i": clip_i, "clip_t": clip_t})

    # Summary
    for arm in ["base", "finetuned"]:
        avg_i = sum(r["clip_i"] for r in results[arm]) / len(results[arm])
        avg_t = sum(r["clip_t"] for r in results[arm]) / len(results[arm])
        results[f"{arm}_summary"] = {"avg_clip_i": avg_i, "avg_clip_t": avg_t}
        print(f"{arm}: avg CLIP-I={avg_i:.4f}, avg CLIP-T={avg_t:.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()