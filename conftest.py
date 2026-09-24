"""Root conftest.py — ensures 'src/' is on sys.path for all test discovery."""

import sys
import os

# Make 'src/xautonet' importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
