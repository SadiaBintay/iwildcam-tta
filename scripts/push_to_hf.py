"""Upload build/webapp to a Hugging Face Space (free CPU).
  python scripts/push_to_hf.py            # asks for your Hugging Face *write* token"""
import argparse
from getpass import getpass
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--folder", default=str(ROOT / "build" / "webapp"))
ap.add_argument("--name", default="camera-trap-adapt")
a = ap.parse_args()
api = HfApi(token=getpass("Hugging Face token (write): "))
repo = f"{api.whoami()['name']}/{a.name}"
api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(folder_path=a.folder, repo_id=repo, repo_type="space", ignore_patterns=["__pycache__"])
print("https://huggingface.co/spaces/" + repo)
