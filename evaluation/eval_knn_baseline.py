import json
import os

import torch
from eval_suite import (
    DEVICE,
    INSTANCE_DATA_DIR,
    PROMPT_SUITE,
    compute_clip_i,
    compute_clip_t,
    load_clip_model,
)
from PIL import Image

RESULTS_PATH = "/root/anchor/evaluation/results_knn.json"


def load_reference_pool():
    pool = [
        os.path.join(INSTANCE_DATA_DIR, f)
        for f in os.listdir(INSTANCE_DATA_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    class_dir = "/root/anchor/checkpoints/class_images/sneaker"
    if os.path.isdir(class_dir):
        class_images = sorted(os.listdir(class_dir))[:20]
        pool += [os.path.join(class_dir, f) for f in class_images]
    return pool


def retrieve_nearest(model, preprocess, tokenizer, prompt, pool_paths):
    with torch.no_grad():
        text_tokens = tokenizer([prompt]).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        best_score = -1.0
        best_path = None
        for path in pool_paths:
            image = Image.open(path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).item()
            if score > best_score:
                best_score = score
                best_path = path

    return best_path


def main():
    reference_images = [
        os.path.join(INSTANCE_DATA_DIR, f)
        for f in os.listdir(INSTANCE_DATA_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    pool_paths = load_reference_pool()

    clip_model, clip_preprocess, clip_tokenizer = load_clip_model()

    results = []
    for prompt in PROMPT_SUITE:
        clean_prompt = prompt.replace("sks ", "")
        retrieved_path = retrieve_nearest(
            clip_model, clip_preprocess, clip_tokenizer, clean_prompt, pool_paths
        )
        retrieved_image = Image.open(retrieved_path).convert("RGB")

        clip_i = compute_clip_i(clip_model, clip_preprocess, retrieved_image, reference_images)
        clip_t = compute_clip_t(clip_model, clip_preprocess, clip_tokenizer, retrieved_image, clean_prompt)
        results.append({
            "prompt": clean_prompt,
            "retrieved_from": retrieved_path,
            "clip_i": clip_i,
            "clip_t": clip_t,
        })

    avg_i = sum(r["clip_i"] for r in results) / len(results)
    avg_t = sum(r["clip_t"] for r in results) / len(results)
    print(f"knn_baseline: avg CLIP-I={avg_i:.4f}, avg CLIP-T={avg_t:.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {"knn_baseline": results, "knn_baseline_summary": {"avg_clip_i": avg_i, "avg_clip_t": avg_t}},
            f, indent=2,
        )


if __name__ == "__main__":
    main()
