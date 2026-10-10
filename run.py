"""
Universal Smart Launcher for Mood-Based Music Recommender.
- Automatically verifies if all required dependencies are installed and compatible.
- Installs missing or incompatible dependencies automatically when needed.
- Verifies that the trained model (model.h5 & labels.npy) exists, or trains it automatically.
- Launches and opens the Mood-Based Music Recommender web application.
"""

import os
import sys
import shutil
import subprocess
import argparse

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

REQUIREMENTS_FILE = os.path.join(PROJECT_DIR, "requirements.txt")
MODEL_FILE = os.path.join(PROJECT_DIR, "model.h5")
LABELS_FILE = os.path.join(PROJECT_DIR, "labels.npy")
APP_FILE = os.path.join(PROJECT_DIR, "app.py")


def print_banner():
    print("=" * 64)
    print("   MOOD-BASED MUSIC RECOMMENDATION SYSTEM - SMART LAUNCHER")
    print("=" * 64)


def check_dependencies() -> bool:
    """Fast check (<0.05s) to verify all required packages and compatible versions are installed."""
    import importlib.metadata
    import importlib.util

    required_modules = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("av", "av"),
        ("streamlit", "streamlit"),
        ("streamlit_webrtc", "streamlit-webrtc"),
        ("mediapipe", "mediapipe"),
        ("tensorflow", "tensorflow"),
    ]

    for mod_name, dist_name in required_modules:
        if importlib.util.find_spec(mod_name) is None:
            print(f"[INFO] Missing required module: '{mod_name}' ({dist_name})")
            return False

    try:
        mp_ver = importlib.metadata.version("mediapipe")
        # mediapipe >= 0.10.30 / 1.0 removed mp.solutions.holistic
        major, minor, *patch = [int(x) for x in mp_ver.split(".")[:3] if x.isdigit()]
        if major >= 1 or (major == 0 and minor == 10 and patch and patch[0] > 21):
            print(f"[INFO] Incompatible MediaPipe version ({mp_ver}). Requires <= 0.10.21.")
            return False
    except Exception:
        return False

    return True


def install_dependencies():
    """Install all required packages from requirements.txt."""
    print("\n[SETUP] Installing required packages... Please wait.")
    uv_bin = shutil.which("uv")

    if uv_bin:
        print("[SETUP] Using 'uv' for fast package installation...")
        cmd = [
            uv_bin,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--link-mode=copy",
            "-r",
            REQUIREMENTS_FILE,
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("[OK] All dependencies installed successfully via uv!\n")
            return

    # Standard pip fallback
    print("[SETUP] Using 'pip' to install requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE]
    )
    print("[OK] All dependencies installed successfully!\n")


def ensure_model_exists():
    """Verify model.h5 and labels.npy exist; if not, run train.py automatically."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        print("\n[SETUP] Model files not found. Training model from 'data/'...")
        subprocess.check_call([sys.executable, os.path.join(PROJECT_DIR, "train.py")])


def launch_streamlit_app():
    """Launch the Streamlit web application and open it in the default browser."""
    print("\n[START] Launching Mood-Based Music Recommender Web App...")
    print("[INFO] The app will open automatically in your web browser at http://localhost:8501")
    print("[INFO] Press Ctrl+C in this window at any time to stop the server.\n")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP_FILE,
        "--server.headless",
        "false",
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Launcher for Mood-Based Music Recommender"
    )
    parser.add_argument(
        "--mode",
        choices=["web", "camera", "train", "collect"],
        default="web",
        help="Execution mode: 'web' (default Streamlit UI), 'camera' (OpenCV window), 'train', or 'collect'",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify/install requirements and model without launching the app.",
    )
    args = parser.parse_args()

    print_banner()

    # Step 1: Verify or install dependencies
    print("[1/3] Checking system requirements...")
    if check_dependencies():
        print("[OK] All requirements are already satisfied!")
    else:
        install_dependencies()
        if not check_dependencies():
            print("[ERROR] Dependency verification failed after installation.")
            sys.exit(1)

    # Step 2: Ensure trained model exists
    print("[2/3] Verifying trained emotion model...")
    ensure_model_exists()
    print("[OK] Emotion classifier model and labels are ready!")

    # Ensure Desktop & Project App Start Icon points to current directory
    if os.name == "nt" and os.path.exists(os.path.join(PROJECT_DIR, "create_shortcut.py")):
        try:
            import create_shortcut
            create_shortcut.main()
        except Exception:
            pass

    if args.check_only:
        print("[3/3] Environment check complete. Ready to run!")
        return

    # Step 3: Run requested mode
    print(f"[3/3] Starting application in '{args.mode}' mode...")
    if args.mode == "web":
        launch_streamlit_app()
    elif args.mode == "camera":
        subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "inference.py")])
    elif args.mode == "train":
        subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "train.py")])
    elif args.mode == "collect":
        subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "data_collection.py")])


if __name__ == "__main__":
    main()
