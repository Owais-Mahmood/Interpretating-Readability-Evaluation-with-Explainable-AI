from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--base", default="configs/base.yaml")
    args = parser.parse_args()
    destination = Path("configs/experiments") / f"{args.name}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(args.base, destination)
    print(f"Created {destination}")


if __name__ == "__main__":
    main()
