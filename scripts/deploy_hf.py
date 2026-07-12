import os
import sys
from huggingface_hub import HfApi

def deploy():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        sys.exit(1)
        
    repo_id = "arunkumar2612/stock-price-prediction-backend"
    print(f"Initiating Hugging Face API connection for space: {repo_id}...")
    
    api = HfApi(token=token)
    
    # Upload the repository folder to the space.
    # The Hugging Face API automatically chunk-uploads large files (like our .pth checkpoints)
    # to LFS storage, resolving git push blockages!
    print("Uploading files and checkpoints (with automatic LFS mapping)...")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".git/*",
            ".github/*",
            "frontend/node_modules/*",
            "frontend/dist/*",
            "data/*.db",
            "__pycache__/*",
            "*.pyc"
        ]
    )
    print("Successfully synchronized workspace to Hugging Face Spaces!")

if __name__ == "__main__":
    deploy()
