from huggingface_hub import snapshot_download


def download_subject(subject: str, local_dir: str = "data/raw") -> str:
    """
    Download a single subject folder from the official Google DreamBooth
    benchmark dataset on Hugging Face.
    """
    path = snapshot_download(
        repo_id="google/dreambooth",
        repo_type="dataset",
        allow_patterns=f"dataset/{subject}/*",
        local_dir=local_dir,
    )
    return path


if __name__ == "__main__":
    download_subject("colorful_sneaker")