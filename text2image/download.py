from huggingface_hub import hf_hub_download

save_path = "models"

file_path = hf_hub_download(
    repo_id="stabilityai/stable-diffusion-3.5-medium",
    filename="sd3.5_medium.safetensors",
    local_dir=save_path
)

print("Model downloaded to:", file_path)
