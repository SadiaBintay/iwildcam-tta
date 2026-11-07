"""Download iWildCam v2.0 (the WILDS version) from a Hugging Face mirror.
Use this ONLY when the official host (CodaLab) fails with 'HTTP Error 404' in build_cache.
The mirror is unofficial (user emmilly); the SHA-256 below is the one its page publishes, so the check only proves
the file matches that page, not that the page is the original authors'. For a course project this is fine; for
publication, cite the official source and verify against WILDS.

  python scripts/download_iwildcam_hf.py --root /content/wilds_data
then
  python -m src.build_cache --root /content/wilds_data --cache-dir CACHE ... --no-download
"""
import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path

REPO, FILE = "emmilly/iwildcam-wilds-v2", "iwildcam_v2.0.tar.gz"
SHA256 = "05c872f46eee44315f79bc90f4098c69ae3799a2b95116db02292ea1b4e370d2"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/wilds_data")
    ap.add_argument("--keep-archive", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    dest = root / "iwildcam_v2.0"
    if (dest / "metadata.csv").exists() and (dest / "RELEASE_v2.0.txt").exists():
        print("already prepared:", dest)
        return
    from huggingface_hub import hf_hub_download
    root.mkdir(parents=True, exist_ok=True)
    print("downloading ~12 GB (10-30 min) ...")
    p = hf_hub_download(REPO, FILE, repo_type="dataset", local_dir=str(root / "_dl"))
    print("checking SHA-256 ...")
    got = sha256(p)
    if got != SHA256:
        raise SystemExit(f"SHA-256 mismatch: {got}\nexpected {SHA256}\nDelete {root/'_dl'} and retry; do not use this file.")
    print("extracting (needs ~12 GB more disk) ...")
    tmp = root / "_x"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    with tarfile.open(p) as t:
        t.extractall(tmp, filter="data")          # 'data' filter blocks paths outside tmp, links, special files
    metas = sorted(tmp.rglob("metadata.csv"), key=lambda q: len(q.parts))
    if not metas:
        raise SystemExit("metadata.csv not found in the archive")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(metas[0].parent), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    (dest / "RELEASE_v2.0.txt").write_text("v2.0\n")
    if not a.keep_archive:
        shutil.rmtree(root / "_dl", ignore_errors=True)
    print("ready:", dest, "| images folder:", (dest / "train").exists() or any(dest.glob("*/")))


if __name__ == "__main__":
    main()
