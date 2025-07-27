"""
download_addons.py
===================

This script automatically downloads additional face‑swap projects into the
`addons` directory.  It clones the following GitHub repositories using a
shallow clone (depth=1) so that you can explore them offline without manually
invoking `git clone` yourself:

* visomaster/VisoMaster
* iVideoGameBoss/iRoopDeepFaceCam
* wladradchenko/wunjo.wladradchenko.ru

Run this script from the root of the `face_swap_windows` package *after* you
have installed Git (the README includes instructions to install Git via
Chocolatey).  Use PowerShell or a CMD prompt:

```cmd
conda activate faceswap
python download_addons.py
```

If the target directory already exists, it will be skipped.  Note that these
repositories are large and the download will take some time depending on your
internet connection.
"""

import subprocess
import sys
from pathlib import Path


REPOS = {
    "VisoMaster": "https://github.com/visomaster/VisoMaster.git",
    "iRoopDeepFaceCam": "https://github.com/iVideoGameBoss/iRoopDeepFaceCam.git",
    "wunjo": "https://github.com/wladradchenko/wunjo.wladradchenko.ru.git",
}


def clone_repo(name: str, url: str, dest: Path):
    target = dest / name
    if target.exists():
        print(f"Repository {name} already exists at {target}, skipping.")
        return
    print(f"Cloning {name} into {target}...")
    cmd = ["git", "clone", "--depth", "1", url, str(target)]
    try:
        subprocess.check_call(cmd)
        print(f"Finished cloning {name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {name}: {e}")


def main():
    addons_dir = Path(__file__).resolve().parent / "addons"
    addons_dir.mkdir(exist_ok=True)
    for name, url in REPOS.items():
        clone_repo(name, url, addons_dir)


if __name__ == "__main__":
    main()