#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.slurm_batch import main_evaluate_batch


if __name__ == "__main__":
    if "--task-id" not in sys.argv and "SLURM_ARRAY_TASK_ID" in os.environ:
        sys.argv.extend(["--task-id", os.environ["SLURM_ARRAY_TASK_ID"]])
    main_evaluate_batch()
