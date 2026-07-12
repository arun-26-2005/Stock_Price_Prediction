import os
import sys
from huggingface_hub import HfApi

def deploy():
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    if not token or not repo_id:
        print("Error: Both HF_TOKEN and HF_REPO_ID environment variables must be set.")
        sys.exit(1)
        
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
