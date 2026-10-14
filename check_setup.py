"""
Run this BEFORE the app:   python check_setup.py
It tests each part separately and prints PASS or FAIL with the fix for the failing part.
"""
import os
import struct
import subprocess
import sys
import tempfile

failed = []


def step(name, fn, hint):
    try:
        msg = fn()
        print(f"[PASS] {name}" + (f"  ({msg})" if msg else ""))
        return True
    except Exception as e:
        text = "\n".join(str(e).splitlines()[-15:])   # the real cause is at the END of the message
        print(f"[FAIL] {name}\n{type(e).__name__}: {text}\n         FIX: {hint}")
        failed.append(name)
        return False


def check_python():
    v = sys.version_info
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("This is 32-bit Python. Install 64-bit Python 3.11 from python.org.")
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} is too old. Use Python 3.11.")
    note = "" if v < (3, 13) else "  (3.11 is the recommended version)"
    return f"Python {v.major}.{v.minor}.{v.micro}{note}"


def check_env():
    exe = os.path.abspath(sys.executable)
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("You are not inside a virtual environment.")
    here = os.path.abspath(os.getcwd()).lower()
    if not exe.lower().startswith(here):
        raise RuntimeError(f"This environment is NOT inside this project folder:\n         {exe}")
    return exe


def check_packages():
    import importlib
    names = ["torch", "torchvision", "torchaudio", "transformers", "einops", "numpy",
             "PIL", "cv2", "streamlit", "imageio_ffmpeg"]
    missing, versions = [], []
    for n in names:
        try:
            m = importlib.import_module(n)
            versions.append(f"{n} {getattr(m, '__version__', '')}".strip())
        except Exception:
            missing.append(n)
    if missing:
        raise ImportError("missing: " + ", ".join(missing))
    return f"{len(names)} packages found"


def check_ffmpeg():
    import detect
    exe = detect.ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found")
    out = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    return out[:50]


def check_voice_model():
    import numpy as np
    import detect
    from transformers import pipeline
    import torch
    pipe = pipeline("antispoofing", model=detect.AUDIO_MODEL, trust_remote_code=True,
                    device="cuda" if torch.cuda.is_available() else "cpu")
    detect._audio_pipe = pipe
    out = pipe(np.random.randn(64600).astype("float32") * 0.1)
    p = detect.spoof_prob(out)
    return f"model answered, fake probability on noise = {p:.2f}"


def check_face_model():
    import visual
    from PIL import Image
    import numpy as np
    visual.load_model()
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    p = visual.score_crops([img])[0]
    return f"model answered, fake probability on noise = {p:.2f}"


def check_end_to_end():
    import detect
    exe = detect.ffmpeg_exe()
    mp4 = os.path.join(tempfile.mkdtemp(), "t.mp4")
    subprocess.run([exe, "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-f", "lavfi",
                    "-i", "sine=frequency=300", "-t", "4", "-shortest", mp4],
                   capture_output=True, check=True)
    r = detect.analyze(mp4)
    return f"full pipeline ran, verdict = {r['verdict']} (a test tone, so the verdict itself does not matter)"


if __name__ == "__main__":
    print("Checking your setup...\n")
    step("1. Python version (64-bit, 3.10+)", check_python,
         "Install 64-bit Python 3.11 from python.org, then recreate the .venv.")
    step("2. Virtual environment inside this folder", check_env,
         "Run:  deactivate   then   py -3.11 -m venv .venv   then   .\\.venv\\Scripts\\Activate.ps1")
    if not step("3. Packages installed", check_packages,
                "Run:  python -m pip install -r requirements.txt   (then run this check again)"):
        print("\nStopping: fix step 3 first, then run   python check_setup.py   again.")
        sys.exit(1)
    step("4. ffmpeg (audio extraction)", check_ffmpeg,
         "Run:  python -m pip install imageio-ffmpeg")
    v_ok = step("5. Voice model downloads and runs", check_voice_model,
                "Needs internet for the first download (about 2 GB). Make sure settings.py has "
                "AUDIO_MODEL = \"Speech-Arena-2025/DF_Arena_500M_V_1\". Send me this message if it fails.")
    f_ok = step("6. Face model downloads and runs", check_face_model,
                "Needs internet for the first download. If it mentions a gated/private repo, accept "
                "the license on its Hugging Face page and run:  huggingface-cli login. "
                "If it names a missing package, pip install that package. Send me this message if it fails.")
    if v_ok and f_ok:
        step("7. Full pipeline on a test video", check_end_to_end,
             "Send me this message.")
    print()
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        print("Send me a screenshot of this output.")
        sys.exit(1)
    print("ALL CHECKS PASSED. Start the app with:   streamlit run app.py")