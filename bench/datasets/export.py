import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    tool = (
        Path(__file__).resolve().parents[2] / "scripts/datasets" / Path(__file__).name
    )
    sys.path.insert(0, str(tool.parent))
    runpy.run_path(str(tool), run_name="__main__")
