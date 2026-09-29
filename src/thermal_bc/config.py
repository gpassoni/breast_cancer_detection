"""Project paths.

`ROOT_DIR` (from the environment or a `.env` file) points at the folder containing `data/`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def root_dir() -> Path:
    load_dotenv()
    return Path(os.getenv("ROOT_DIR", "."))
