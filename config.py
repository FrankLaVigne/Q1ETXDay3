import os
from pathlib import Path

# Load .env
env_path = Path(__file__).parent / ".env"
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

# Expose as attributes
API_KEY = os.environ["API_KEY"]
ENDPOINT_BASE = os.environ["ENDPOINT_BASE"]