"""
Agentic DermaGuard - Root Streamlit Runner
Delegates execution to DermaGuard/app.py
"""
import os
import sys
from pathlib import Path

# Add DermaGuard folder to Python sys.path
BASE_DIR = Path(__file__).resolve().parent
DERMAGUARD_DIR = BASE_DIR / "DermaGuard"
if str(DERMAGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(DERMAGUARD_DIR))

# Execute app.py
app_path = DERMAGUARD_DIR / "app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, str(app_path), "exec"))
