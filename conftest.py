"""Ensures the project root is importable so `import config`, `import schemas`,
and `import src.<module>` resolve when running `pytest` from anywhere."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
