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

rwgs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rwgs")
os.chdir(rwgs_dir)

try:
    from voila.app import Voila
    v = Voila()
    v.initialize([
        "rwgs_web.ipynb",
        "--Voila.ip=0.0.0.0",
        "--port=7860",
        "--no-browser",
        "--theme=light",
    ])
    v.tornado_settings = {
        "disable_check_xsrf": True,
        "xsrf_cookies": False,
        "headers": {
            "Content-Security-Policy": "frame-ancestors * 'self' https://huggingface.co https://*.hf.space;",
        },
    }
    v.start()
except Exception as e:
    print(f"Direct launch note ({e}), falling back to CLI subprocess...")
    cmd = [
        sys.executable,
        "-m",
        "voila",
        "rwgs_web.ipynb",
        "--Voila.ip=0.0.0.0",
        "--port=7860",
        "--no-browser",
        "--theme=light",
    ]
    subprocess.run(cmd)
