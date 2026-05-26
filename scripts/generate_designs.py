#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.slurm_batch import main_generate


if __name__ == "__main__":
    main_generate()
