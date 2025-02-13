from huggingface_hub import snapshot_download

model_name = "cl-nagoya/ruri-large"
download_path = snapshot_download(
    repo_id=model_name,
    local_dir = f"embedding_model/{model_name}",
    local_dir_use_symlinks=False # ※1
    )