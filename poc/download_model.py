#!/usr/bin/env python3
"""Download the model with SSL verification disabled if needed.

Checks if the model exists locally first, only downloads if missing.
"""
import ssl
import os
import sys
from pathlib import Path

# Disable SSL verification before any imports
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

import warnings
warnings.filterwarnings('ignore')

# Aggressively patch requests to disable SSL at the lowest level
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

class NoSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['cert_reqs'] = ssl.CERT_NONE
        kwargs['assert_hostname'] = False
        return super().init_poolmanager(*args, **kwargs)

# Monkey-patch requests.Session to always use our adapter
_original_session_init = requests.Session.__init__
def patched_session_init(self, *args, **kwargs):
    _original_session_init(self, *args, **kwargs)
    self.mount('https://', NoSSLAdapter())
    self.mount('http://', HTTPAdapter())
    self.verify = False
requests.Session.__init__ = patched_session_init

# Now import after SSL is disabled
from huggingface_hub import snapshot_download, try_to_load_from_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def check_model_cached(model_name):
    """Check if model is already cached locally."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"

    if model_dir.exists():
        # Check if it has the essential files
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            return True
    return False

def download_model(model_name):
    """Download the model with progress indication."""
    print(f"\n📦 Downloading {model_name}...")
    print("   This is ~1GB and only happens once.")
    print("   ⚠️  SSL verification is disabled for this download.\n")

    try:
        # Download model files
        snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.gguf", "*.bin"],  # Skip unnecessary formats
        )

        print("\n✅ Model downloaded successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def verify_model_loads(model_name):
    """Quick check that the model can be loaded."""
    print(f"\n🔍 Verifying {model_name} can be loaded...")
    try:
        # Just load tokenizer as a quick check
        AutoTokenizer.from_pretrained(model_name)
        print("✅ Model verified and ready to use!")
        return True
    except Exception as e:
        print(f"⚠️  Model verification failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("minijev Model Setup")
    print("=" * 60)

    if check_model_cached(MODEL_NAME):
        print(f"\n✅ {MODEL_NAME} is already cached locally.")
        verify_model_loads(MODEL_NAME)
    else:
        print(f"\n⚠️  {MODEL_NAME} not found locally.")
        if download_model(MODEL_NAME):
            verify_model_loads(MODEL_NAME)
        else:
            print("\n❌ Failed to download model. Check your network connection.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Model setup complete. You can now start the server.")
    print("=" * 60)
