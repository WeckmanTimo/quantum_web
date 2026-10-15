"""
Comp-Chem-Lab: Hugging Face Spaces Entrypoint
===========================================
Launches the Voilà web dashboard on port 7860 for Hugging Face Spaces.
"""
import os
import subprocess
import sys

print("==================================================")
print(" Starting Comp-Chem-Lab Studio (Voilà on port 7860)")
print("==================================================")

cmd = [
    sys.executable,
    "-m",
    "voila",
    "rwgs_web.ipynb",
    "--port=7860",
    "--no-browser",
    "--theme=light",
    "--enable_nbextensions=True",
]

rwgs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rwgs")
subprocess.run(cmd, cwd=rwgs_dir)

