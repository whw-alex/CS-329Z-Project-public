"""Download the full conversations.parquet (6.4 GB) into data/raw/ via hf_hub_download (resumable, parallel)."""
from pathlib import Path

from huggingface_hub import hf_hub_download

out = Path(__file__).resolve().parent.parent / "data" / "raw"
p = hf_hub_download("SALT-NLP/SWE-chat", "conversations.parquet", repo_type="dataset", local_dir=out)
print("done:", p)
