#!/usr/bin/env python3
"""Start the minijev API server with SSL verification disabled.

This is needed when behind a corporate proxy with self-signed certificates.
Only use this for local development.

Usage:
    uv run python start_with_no_ssl.py
"""
import ssl
import os
import sys
from pathlib import Path

# Disable SSL verification BEFORE any imports that use SSL
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

import warnings
warnings.filterwarnings('ignore')

# Aggressively patch requests to disable SSL
import requests
from requests.adapters import HTTPAdapter

class NoSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['cert_reqs'] = ssl.CERT_NONE
        kwargs['assert_hostname'] = False
        return super().init_poolmanager(*args, **kwargs)

_original_session_init = requests.Session.__init__
def patched_session_init(self, *args, **kwargs):
    _original_session_init(self, *args, **kwargs)
    self.mount('https://', NoSSLAdapter())
    self.mount('http://', HTTPAdapter())
    self.verify = False
requests.Session.__init__ = patched_session_init

# Check if model exists, download if not
def check_model_exists():
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / "models--Qwen--Qwen2.5-0.5B-Instruct"
    return model_dir.exists()

if not check_model_exists():
    print("⚠️  Model not found. Running download script first...")
    import subprocess
    result = subprocess.run([sys.executable, "download_model.py"], cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("❌ Failed to download model. Exiting.")
        sys.exit(1)
    print()

# Start the server
import uvicorn
from server import app

if __name__ == "__main__":
    print("Starting minijev API with SSL verification disabled...")
    print("⚠️  SSL verification is OFF - use only for local development")
    print("Server: http://127.0.0.1:8000")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8000)
