# colab_setup.py

import os
import subprocess

REPO = "transd-levelset"
BRANCH = "sw_wa"

if not os.path.isdir(REPO):
    subprocess.run(
        [
            "git", "clone",
            "--depth", "1",
            "--branch", BRANCH,
            f"https://github.com/jgeomos/{REPO}.git"
        ],
        check=True
    )
print(f"Github repository {REPO}, branch {BRANCH}: successfully cloned.")

if os.path.basename(os.getcwd()) != REPO:
    os.chdir(REPO)

subprocess.run(
    ["pip", "install", "-q",
     "scikit-fmm",
     "connected-components-3d",
     "colorcet",
     "vtk"],
    check=True
)

print("Dependencies: installed.")