"""
Unified Mood-Based Music Recommendation System (Smart Launcher + Web App).
- If run via 'python app.py' (or via the 'Mood Music Recommender' app icon):
  1. Checks if all required dependencies are installed and compatible.
  2. Automatically installs missing requirements if needed.
  3. Verifies trained model files (model.h5 & labels.npy) or trains them automatically.
  4. Launches and opens the interactive Streamlit web application in the browser.
- Inside the Streamlit Web Application:
  1. Real-time webcam mood detection (Happy, Sad, Angry, Surprised, Shocked, Romantic, Energy, Not OK, Neutral)
     combining the trained Keras Neural Network (model.h5) with Scale-Invariant Facial Action Units (FAUs) & Hand Gestures.
  2. Persistent User History (data/user_history.json) tracking the user's favorite Actors, Singers, Languages, and Moods.
  3. Personalized Song Recommendations based on Live Webcam Mood + Past Actor & Language History.
"""

import collections
import datetime
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.parse
import webbrowser

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

REQUIREMENTS_FILE = os.path.join(PROJECT_DIR, "requirements.txt")
MODEL_FILE = os.path.join(PROJECT_DIR, "model.h5")
LABELS_FILE = os.path.join(PROJECT_DIR, "labels.npy")
HISTORY_FILE = os.path.join(PROJECT_DIR, "data", "user_history.json")


# =====================================================================
# PART 1: SMART SELF-INSTALLING BOOTSTRAPPER (WHEN RUN DIRECTLY)
# =====================================================================
def check_dependencies() -> bool:
    """Fast check (<0.05s) to verify all required packages and compatible versions are installed."""
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
            print(f"[INFO] Missing required package: {dist_name}")
            return False

    try:
        mp_ver = importlib.metadata.version("mediapipe")
        major, minor, *patch = [int(x) for x in mp_ver.split(".")[:3] if x.isdigit()]
        if major >= 1 or (major == 0 and minor == 10 and patch and patch[0] > 21):
            print(f"[INFO] Incompatible MediaPipe version ({mp_ver}). Requires <= 0.10.21.")
            return False
    except Exception:
        return False

    return True


def install_dependencies():
    """Automatically install all required packages from requirements.txt."""
    print("\n[SETUP] Installing required packages... Please wait.")
    uv_bin = shutil.which("uv")
    if uv_bin:
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
        if subprocess.run(cmd).returncode == 0:
            print("[OK] All dependencies installed successfully!\n")
            return

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
    print("[OK] All dependencies installed successfully!\n")


def ensure_shortcut_exists():
    """Ensure the single 'Mood Music Recommender' shortcut icon points to assets/launch_helper.bat."""
    if os.name != "nt":
        return
    helper_bat = os.path.join(PROJECT_DIR, "assets", "launch_helper.bat")
    icon_ico = os.path.join(PROJECT_DIR, "assets", "app_icon.ico")
    if not os.path.exists(helper_bat) or not os.path.exists(icon_ico):
        return

    targets = [os.path.join(PROJECT_DIR, "Mood Music Recommender.lnk")]
    parent_dir = os.path.dirname(PROJECT_DIR)
    if os.path.exists(parent_dir) and "desktop" in parent_dir.lower():
        targets.append(os.path.join(parent_dir, "Mood Music Recommender.lnk"))

    for lnk_path in targets:
        ps_cmd = (
            f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk_path}');"
            f"$s.TargetPath='{helper_bat}';"
            f"$s.WorkingDirectory='{PROJECT_DIR}';"
            f"$s.IconLocation='{icon_ico},0';"
            f"$s.Description='Launch Mood-Based Music Recommender';"
            f"$s.Save()"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True,
        )


def is_running_in_streamlit() -> bool:
    """Return True if currently executing inside a Streamlit script runner context."""
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def bootstrap_and_launch():
    """Entry point when app.py is run directly via Python or the App Start Icon."""
    print("=" * 64)
    print("   MOOD-BASED MUSIC RECOMMENDATION SYSTEM")
    print("=" * 64)

    if not check_dependencies():
        install_dependencies()
        if not check_dependencies():
            print("[ERROR] Could not verify dependencies after installation.")
            sys.exit(1)
    else:
        print("[OK] All requirements satisfied!")

    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        print("[SETUP] Training emotion model from data/...")
        subprocess.check_call([sys.executable, os.path.join(PROJECT_DIR, "train.py")])

    ensure_shortcut_exists()

    if "--check-only" in sys.argv:
        print("[OK] Environment, model, and App Start Icon verified!")
        sys.exit(0)

    print("[START] Opening Mood Music Recommender in your browser...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            os.path.abspath(__file__),
            "--server.headless",
            "false",
        ]
    )
    sys.exit(0)


if __name__ == "__main__" and not is_running_in_streamlit():
    bootstrap_and_launch()


# =====================================================================
# PART 2: STREAMLIT WEB APPLICATION & HYBRID EMOTION ENGINE
# =====================================================================
import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

st.set_page_config(
    page_title="Mood Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .hero-banner {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a11cb 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }
    .hero-banner h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
    .hero-banner p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1.02rem; }
    .mood-badge {
        padding: 1.1rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #fff;
        text-align: center;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    .history-card {
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        background: rgba(106, 17, 203, 0.08);
        border-left: 4px solid #6a11cb;
        margin-bottom: 0.75rem;
    }
    .stButton>button {
        border-radius: 10px;
        height: 3.0em;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-banner">
    <h1>🎵 Mood-Based Music Recommendation System</h1>
    <p>Real-Time Webcam Emotion Recognition + Personalized Recommendations from Your Actor &amp; Language History</p>
</div>
""",
    unsafe_allow_html=True,
)

MOOD_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "energy": "⚡",
    "surprised": "😮",
    "shocked": "😱",
    "angry": "😠",
    "romantic": "❤️",
    "not_ok": "😔",
    "neutral": "😐",
}

ALL_MOODS = [
    "happy",
    "sad",
    "energy",
    "romantic",
    "surprised",
    "shocked",
    "angry",
    "not_ok",
    "neutral",
]

POPULAR_ACTORS_BY_LANG = {
    "Hindi": [
        "Shah Rukh Khan",
        "Arijit Singh",
        "Salman Khan",
        "Ranbir Kapoor",
        "Hrithik Roshan",
        "Kartik Aaryan",
        "Akshay Kumar",
        "Shreya Ghoshal",
        "Atif Aslam",
        "Kishore Kumar",
    ],
    "Tamil": [
        "Thalapathy Vijay",
        "Anirudh Ravichander",
        "Ajith Kumar",
        "Rajinikanth",
        "Suriya",
        "Dhanush",
        "Kamal Haasan",
        "A.R. Rahman",
        "Sid Sriram",
    ],
    "Telugu": [
        "Allu Arjun",
        "Prabhas",
        "Mahesh Babu",
        "Ram Charan",
        "Jr NTR",
        "Pawan Kalyan",
        "Vijay Deverakonda",
        "Sid Sriram",
        "Devi Sri Prasad",
    ],
    "Punjabi": [
        "Diljit Dosanjh",
        "Sidhu Moose Wala",
        "AP Dhillon",
        "Karan Aujla",
        "Guru Randhawa",
        "B Praak",
        "Amrinder Gill",
    ],
    "English": [
        "The Weeknd",
        "Ed Sheeran",
        "Taylor Swift",
        "Bruno Mars",
        "Eminem",
        "Coldplay",
        "Imagine Dragons",
        "Justin Bieber",
        "Dua Lipa",
    ],
    "Kannada": [
        "Yash",
        "Puneeth Rajkumar",
        "Darshan",
        "Sudeep",
        "Rishab Shetty",
        "Sanjith Hegde",
    ],
    "Malayalam": [
        "Dulquer Salmaan",
        "Mohanlal",
        "Mammootty",
        "Fahadh Faasil",
        "Prithviraj Sukumaran",
        "Vineeth Sreenivasan",
    ],
    "Spanish": ["Bad Bunny", "Shakira", "Enrique Iglesias", "J Balvin", "Rosalia"],
    "Korean": ["BTS", "BLACKPINK", "IU", "Stray Kids", "NewJeans", "EXO"],
}


# =====================================================================
# PART 3: PERSISTENT USER HISTORY (ACTOR / SINGER & LANGUAGE)
# =====================================================================
def load_user_history() -> dict:
    """Load persistent user history of actors, languages, and mood sessions."""
    default_data = {
        "last_language": "Hindi",
        "last_actor": "Shah Rukh Khan",
        "last_platform": "YouTube",
        "favorite_actors": ["Shah Rukh Khan", "Arijit Singh", "Thalapathy Vijay", "Allu Arjun"],
        "favorite_languages": ["Hindi", "Tamil", "Telugu", "English", "Punjabi"],
        "history": [],
    }
    if not os.path.exists(HISTORY_FILE):
        return default_data
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in default_data.items():
                data.setdefault(k, v)
            return data
    except Exception:
        return default_data


def save_user_history(language: str, actor: str, mood: str, platform: str, query: str, url: str):
    """Save the user's selected Actor/Singer, Language, and detected Mood to persistent history."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    data = load_user_history()

    language = language.strip() or "Hindi"
    actor = actor.strip()

    data["last_language"] = language
    if actor:
        data["last_actor"] = actor
    data["last_platform"] = platform

    # Update ranked favorite actors & languages (most recent first, deduplicated)
    if actor:
        fav_actors = [a for a in data.get("favorite_actors", []) if a.lower() != actor.lower()]
        fav_actors.insert(0, actor)
        data["favorite_actors"] = fav_actors[:15]

    fav_langs = [l for l in data.get("favorite_languages", []) if l.lower() != language.lower()]
    fav_langs.insert(0, language)
    data["favorite_languages"] = fav_langs[:10]

    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mood": mood,
        "language": language,
        "actor": actor or "Any Artist",
        "platform": platform,
        "query": query,
        "url": url,
    }
    history_list = data.get("history", [])
    history_list.insert(0, entry)
    data["history"] = history_list[:30]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =====================================================================
# PART 4: SINGLETON MODEL & REAL-TIME EMOTION DETECTION ENGINE
# =====================================================================
@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        return None, None
    loaded_model = load_model(MODEL_FILE, compile=False)
    loaded_labels = np.load(LABELS_FILE)
    return loaded_model, loaded_labels


model, labels = load_emotion_model()


class SharedEmotionTracker:
    """Process-wide thread-safe emotion tracker shared between WebRTC thread and Streamlit UI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.recent_moods = collections.deque(maxlen=6)
        self.current_mood = ""
        self.confidence = 0.0
        self.expression_detail = ""
        self.mood_counts = collections.Counter()

    def record_prediction(self, mood: str, confidence: float, detail: str = ""):
        with self.lock:
            self.recent_moods.append(mood)
            # Majority vote over the last 6 frames for smooth, flicker-free detection
            most_common_mood, _ = collections.Counter(self.recent_moods).most_common(1)[0]
            self.current_mood = most_common_mood
            self.confidence = confidence
            self.expression_detail = detail
            self.mood_counts[most_common_mood] += 1

    def set_direct(self, mood: str, confidence: float, detail: str = ""):
        with self.lock:
            self.recent_moods.clear()
            self.recent_moods.append(mood)
            self.current_mood = mood
            self.confidence = confidence
            self.expression_detail = detail
            self.mood_counts[mood] += 1

    def get_status(self):
        with self.lock:
            return self.current_mood, self.confidence, self.expression_detail


@st.cache_resource
def get_emotion_tracker() -> SharedEmotionTracker:
    return SharedEmotionTracker()


emotion_tracker = get_emotion_tracker()

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def classify_emotion_hybrid(results) -> tuple[str, float, str]:
    """
    Hybrid Emotion Classifier combining:
    1. Trained Deep Neural Network (model.h5) using exact nose/hand-relative landmarks
    2. Calibrated Facial Action Units (FAUs) & Hand Gesture Geometry for real-time webcam responsiveness
    Detects: happy, sad, energy, not_ok, shocked, surprised, romantic, angry, neutral.
    """
    if not results.face_landmarks:
        return "", 0.0, "No face detected"

    lm = results.face_landmarks.landmark
    has_lh = results.left_hand_landmarks is not None
    has_rh = results.right_hand_landmarks is not None

    # --- 1. Exact Relative Feature Vector for Trained Neural Network (model.h5) ---
    nn_mood = "neutral"
    nn_conf = 0.0
    probs_dict = {}

    if model is not None and labels is not None:
        lst = []
        ref_x, ref_y = lm[1].x, lm[1].y
        for p in lm:
            lst.append(p.x - ref_x)
            lst.append(p.y - ref_y)

        if has_lh:
            lh_lm = results.left_hand_landmarks.landmark
            lh_x, lh_y = lh_lm[8].x, lh_lm[8].y
            for p in lh_lm:
                lst.append(p.x - lh_x)
                lst.append(p.y - lh_y)
        else:
            lst.extend([0.0] * 42)

        if has_rh:
            rh_lm = results.right_hand_landmarks.landmark
            rh_x, rh_y = rh_lm[8].x, rh_lm[8].y
            for p in rh_lm:
                lst.append(p.x - rh_x)
                lst.append(p.y - rh_y)
        else:
            lst.extend([0.0] * 42)

        try:
            features = np.array(lst, dtype=np.float32).reshape(1, -1)
            preds = model.predict(features, verbose=0)[0]
            probs_dict = {str(labels[i]): float(preds[i]) for i in range(len(labels))}
            idx = int(np.argmax(preds))
            nn_conf = float(preds[idx]) * 100.0
            if idx < len(labels):
                nn_mood = str(labels[idx])
        except Exception:
            pass

    # --- 2. Compute Calibrated Scale-Invariant Facial Action Units (FAUs) ---
    face_w = max(abs(lm[454].x - lm[234].x), 1e-4)
    face_h = max(abs(lm[152].y - lm[10].y), 1e-4)

    lip_center_y = (lm[13].y + lm[14].y) / 2.0
    corner_avg_y = (lm[61].y + lm[291].y) / 2.0
    smile_lift = (lip_center_y - corner_avg_y) / face_h
    mouth_w_ratio = abs(lm[291].x - lm[61].x) / face_w
    mouth_open_ratio = abs(lm[14].y - lm[13].y) / face_h
    brow_lift_ratio = ((lm[159].y - lm[105].y) + (lm[386].y - lm[334].y)) / (2.0 * face_h)
    head_tilt = abs(lm[454].y - lm[234].y) / face_w

    # --- 3. Fuse Neural Network + Real-Time Expression Geometry ---
    # A. Wide open jaw -> Shocked
    if mouth_open_ratio > 0.125:
        return "shocked", max(nn_conf, 95.0), "Wide open mouth (shocked)"

    # B. Clear smile -> Happy (or Romantic if strong head tilt)
    if (mouth_w_ratio > 0.422 and smile_lift > 0.012) or (smile_lift > 0.018 and mouth_open_ratio < 0.11):
        if head_tilt > 0.115 and not has_lh and not has_rh:
            return "romantic", 92.0, "Warm smile + head tilt"
        return "happy", max(nn_conf, 94.0), "Smiling expression"

    # C. Trust high-confidence Neural Network prediction on trained classes
    if nn_conf >= 65.0:
        if nn_mood == "not_ok" and nn_conf < 82.0 and probs_dict.get("surprised", 0.0) > 0.20:
            return "surprised", 86.0, "AI Model + Landmarks (surprised)"
        # If hands are down and person is frowning strongly on webcam, allow sad/angry detection
        if not has_lh and not has_rh:
            if brow_lift_ratio < 0.095 and smile_lift < -0.005 and mouth_open_ratio < 0.03:
                return "angry", 88.0, "Furrowed brows + tense mouth"
            if smile_lift < -0.004 and nn_mood not in ("sad", "not_ok"):
                return "sad", 89.0, "Downturned mouth corners (sad)"
        return nn_mood, nn_conf, f"AI Model + Landmarks ({nn_mood})"

    # D. Fallback geometry when NN confidence is lower (e.g. different camera distance)
    if has_lh and has_rh:
        return "energy", 90.0, "Both hands active"
    if smile_lift > 0.012:
        return "happy", 85.0, "Gentle smile"
    if smile_lift < 0.002 or brow_lift_ratio < 0.112:
        return "sad", 84.0, "Sad / serious expression"
    if mouth_open_ratio > 0.065:
        return "surprised", 85.0, "Surprised open mouth"

    return nn_mood or "neutral", max(nn_conf, 75.0), "Detected from facial landmarks"


def draw_styled_landmarks(frame, results, mood: str = "", conf: float = 0.0, detail: str = ""):
    """Draw face/hand landmarks and a sleek status overlay banner on the video frame."""
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 140), thickness=1),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 90, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 130, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
        )

    if mood:
        # Draw dark semi-transparent HUD bar at top of frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (460, 82), (20, 20, 35), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        cv2.putText(
            frame,
            f"MOOD: {mood.upper()} ({conf:.0f}%)",
            (24, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (56, 239, 125),
            2,
            cv2.LINE_AA,
        )
        if detail:
            cv2.putText(
                frame,
                detail[:48],
                (24, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 240),
                1,
                cv2.LINE_AA,
            )


RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class EmotionVideoProcessor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        mood, conf, detail = classify_emotion_hybrid(results)
        if mood:
            emotion_tracker.record_prediction(mood, conf, detail)
            curr_mood, curr_conf, curr_detail = emotion_tracker.get_status()
            draw_styled_landmarks(frm, results, curr_mood, curr_conf, curr_detail)
        else:
            draw_styled_landmarks(frm, results)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# =====================================================================
# PART 5: USER INTERFACE (WEBCAM + PAST ACTOR/LANGUAGE RECOMMENDATIONS)
# =====================================================================
user_hist = load_user_history()

with st.sidebar:
    st.header("👤 Your Music Profile")
    st.caption("Automatically remembered from your past listening history:")
    st.markdown(f"- **Last Language**: `{user_hist['last_language']}`")
    st.markdown(f"- **Last Actor / Singer**: `{user_hist['last_actor']}`")
    if user_hist.get("favorite_actors"):
        st.markdown(
            "- **Recent Actors/Singers**: "
            + ", ".join([f"`{a}`" for a in user_hist["favorite_actors"][:5]])
        )

    st.markdown("---")
    st.subheader("🎭 Detectable Webcam Moods")
    st.write(
        " • ".join([f"{MOOD_EMOJIS.get(m, '🎵')} `{m.title()}`" for m in ALL_MOODS])
    )
    st.caption(
        "Try smiling (**Happy**), frowning/looking down (**Sad**), dropping jaw (**Shocked**), raising eyebrows (**Surprised**), lowering brows (**Angry**), tilting head with a smile (**Romantic**), or raising hands (**Energy**)!"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Past User History", use_container_width=True):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("User history cleared!")
        st.rerun()


col_cam, col_rec = st.columns([1.25, 1.0], gap="large")

with col_cam:
    st.subheader("📹 Step 1: Detect Your Mood from Webcam")
    cam_mode = st.radio(
        "Select Webcam Mode:",
        [
            "🎥 Live Webcam Stream (Real-Time Continuous Detection)",
            "📸 Quick Webcam Snapshot (Instant Photo Capture)",
            "🖐️ Manual Mood Selection",
        ],
        index=0,
        horizontal=True,
    )

    if cam_mode.startswith("🎥"):
        st.caption(
            "Click **START** below. Look at the webcam and make any expression (**Happy, Sad, Angry, Surprised, Shocked, Romantic, Energy**) — the panel on the right updates live!"
        )
        webrtc_streamer(
            key="emotion-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=EmotionVideoProcessor,
            async_processing=True,
        )

    elif cam_mode.startswith("📸"):
        snap_file = st.camera_input("Pose in front of the webcam & click Take Photo:")
        if snap_file is not None:
            file_bytes = np.asarray(bytearray(snap_file.read()), dtype=np.uint8)
            bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            with mp_holistic.Holistic(
                static_image_mode=True, min_detection_confidence=0.5
            ) as holistic:
                res = holistic.process(rgb_img)
            mood, conf, detail = classify_emotion_hybrid(res)
            annotated = bgr_img.copy()
            draw_styled_landmarks(annotated, res, mood, conf, detail)
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Detected: {mood.upper() if mood else 'None'} ({conf:.0f}%) - {detail}",
                use_container_width=True,
            )
            if mood:
                emotion_tracker.set_direct(mood, conf, detail)
            else:
                st.warning("Face not clearly visible. Please center your face in good lighting.")
    else:
        st.info("Select your mood manually in the panel on the right.")


with col_rec:
    st.subheader("🎬 Step 2: Language & Actor / Singer (With Past History)")

    # 1. Language Selection (pre-selected from user's past history!)
    base_languages = [
        "Hindi",
        "Tamil",
        "Telugu",
        "Punjabi",
        "English",
        "Kannada",
        "Malayalam",
        "Spanish",
        "Korean",
        "Other",
    ]
    # Put user's past favorite languages at the top
    ordered_langs = []
    for fav_l in user_hist.get("favorite_languages", []):
        if fav_l not in ordered_langs and fav_l != "Other":
            ordered_langs.append(fav_l)
    for bl in base_languages:
        if bl not in ordered_langs:
            ordered_langs.append(bl)

    default_lang_idx = (
        ordered_langs.index(user_hist["last_language"])
        if user_hist["last_language"] in ordered_langs
        else 0
    )
    selected_lang = st.selectbox(
        "🌐 Language (Auto-loaded from your history)",
        ordered_langs,
        index=default_lang_idx,
    )
    if selected_lang == "Other":
        lang = st.text_input("Enter custom language:", value=user_hist["last_language"]).strip()
    else:
        lang = selected_lang

    # 2. Actor / Singer Selection (combining User's Past History + Popular Stars for selected Language + Custom input!)
    lang_stars = POPULAR_ACTORS_BY_LANG.get(lang, POPULAR_ACTORS_BY_LANG["English"])
    actor_options = []

    # First include the user's past history actors so they are always remembered at the top
    for past_a in user_hist.get("favorite_actors", []):
        if past_a and past_a not in actor_options:
            actor_options.append(past_a)
    for star in lang_stars:
        if star not in actor_options:
            actor_options.append(star)
    actor_options.append("✍️ Type Custom Actor / Singer...")
    actor_options.append("Any Actor / All Hits")

    default_actor_idx = (
        actor_options.index(user_hist["last_actor"])
        if user_hist.get("last_actor") in actor_options
        else 0
    )
    selected_actor_choice = st.selectbox(
        "🌟 Actor / Singer (Remembered from your past history + top stars)",
        actor_options,
        index=default_actor_idx,
    )

    if selected_actor_choice == "✍️ Type Custom Actor / Singer...":
        actor = st.text_input(
            "Enter Actor or Singer Name:",
            value=user_hist.get("last_actor", ""),
            placeholder="e.g. Shah Rukh Khan, Thalapathy Vijay, Allu Arjun, Arijit Singh",
        ).strip()
    elif selected_actor_choice == "Any Actor / All Hits":
        actor = ""
    else:
        actor = selected_actor_choice

    platform = st.selectbox(
        "🎧 Music Platform",
        ["YouTube", "YouTube Music", "Spotify"],
        index=["YouTube", "YouTube Music", "Spotify"].index(
            user_hist.get("last_platform", "YouTube")
        )
        if user_hist.get("last_platform") in ["YouTube", "YouTube Music", "Spotify"]
        else 0,
    )

    st.markdown("---")

    # Live Auto-Updating Fragment so webcam mood changes appear automatically every 1.5s!
    @st.fragment(run_every=1.5)
    def live_mood_and_recommendation_panel(chosen_lang: str, chosen_actor: str, chosen_platform: str, mode_str: str):
        live_mood, live_conf, live_detail = emotion_tracker.get_status()

        st.subheader("🎯 Step 3: Live Mood & Personalized Song Recommendations")

        if live_mood and not mode_str.startswith("🖐️"):
            emoji = MOOD_EMOJIS.get(live_mood, "🎵")
            st.markdown(
                f'<div class="mood-badge">{emoji} Webcam Mood: {live_mood.upper()} ({live_conf:.0f}%)<br>'
                f'<span style="font-size:0.85rem;font-weight:500;opacity:0.92;">{live_detail}</span></div>',
                unsafe_allow_html=True,
            )
            override_opts = [f"Use Webcam Mood ({live_mood.title()})"] + [
                m.title() for m in ALL_MOODS
            ]
            override_choice = st.selectbox(
                "Active Mood (Auto-synced from webcam, or pick manually):",
                override_opts,
                index=0,
            )
            if override_choice.startswith("Use Webcam Mood"):
                active_mood = live_mood
            else:
                active_mood = override_choice.lower()
        else:
            if not mode_str.startswith("🖐️"):
                st.info(
                    "👋 Waiting for webcam face detection... (Or pick a mood below anytime)"
                )
            manual_choice = st.selectbox(
                "Select Your Current Mood:",
                [f"{MOOD_EMOJIS.get(m, '🎵')} {m.title()}" for m in ALL_MOODS],
                index=0,
            )
            active_mood = manual_choice.split(" ", 1)[1].lower()

        # Map each mood to natural music search keywords
        mood_keywords = {
            "happy": "happy upbeat dance feel good",
            "sad": "sad emotional heartbreak melody",
            "energy": "high energy workout mass beats",
            "romantic": "romantic love melody hits",
            "surprised": "trending viral party hits",
            "shocked": "high bass energetic electronic",
            "angry": "intense rock rap power",
            "not_ok": "soothing calm relaxing healing",
            "neutral": "chill lofi acoustic evergreens",
        }
        mood_kw = mood_keywords.get(active_mood, active_mood)

        # Build primary query using Past History Actor + Language + Detected Mood
        parts = [chosen_lang]
        if chosen_actor:
            parts.append(chosen_actor)
        parts.extend([mood_kw, "songs"])
        main_query = " ".join(parts)
        enc_main = urllib.parse.quote_plus(main_query)

        yt_url = f"https://www.youtube.com/results?search_query={enc_main}"
        ytm_url = f"https://music.youtube.com/search?q={enc_main}"
        sp_url = f"https://open.spotify.com/search/{urllib.parse.quote(main_query)}"

        platform_map = {
            "YouTube": yt_url,
            "YouTube Music": ytm_url,
            "Spotify": sp_url,
        }
        target_url = platform_map.get(chosen_platform, yt_url)

        actor_display = chosen_actor if chosen_actor else f"Top {chosen_lang} Artists"
        emoji = MOOD_EMOJIS.get(active_mood, "🎵")

        st.markdown(
            f"""
<div class="history-card">
    <strong>✨ Recommended For You (Mood + Actor &amp; Language History):</strong><br>
    {emoji} <b>Mood:</b> {active_mood.title()} &nbsp;|&nbsp;
    🌟 <b>Actor/Singer:</b> {actor_display} &nbsp;|&nbsp;
    🌐 <b>Language:</b> {chosen_lang}
</div>
""",
            unsafe_allow_html=True,
        )

        if st.button(
            f"🎵 Recommend & Open {chosen_lang} {active_mood.title()} Songs ({actor_display})!",
            type="primary",
            use_container_width=True,
        ):
            save_user_history(
                chosen_lang, chosen_actor, active_mood, chosen_platform, main_query, target_url
            )
            st.success(
                f"Saved to history & opening **{active_mood.title()}** songs ft. **{actor_display}** in **{chosen_lang}**!"
            )
            try:
                webbrowser.open_new_tab(target_url)
            except Exception:
                pass

        # Curated multi-variation playlists based on Actor + Language + Mood
        st.markdown("##### 🎶 Instant Curated Playlists (1-Click Open)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.link_button("▶️ YouTube Video Hits", yt_url, use_container_width=True)
        with c2:
            st.link_button("🎧 YouTube Music", ytm_url, use_container_width=True)
        with c3:
            st.link_button("🟢 Spotify Tracks", sp_url, use_container_width=True)

        if chosen_actor:
            jukebox_q = urllib.parse.quote_plus(
                f"{chosen_actor} {chosen_lang} {active_mood} songs jukebox best hits"
            )
            latest_q = urllib.parse.quote_plus(
                f"Best {chosen_lang} {active_mood} songs {chosen_actor} playlist"
            )
            jb_col1, jb_col2 = st.columns(2)
            with jb_col1:
                st.link_button(
                    f"💿 {chosen_actor} {active_mood.title()} Jukebox",
                    f"https://www.youtube.com/results?search_query={jukebox_q}",
                    use_container_width=True,
                )
            with jb_col2:
                st.link_button(
                    f"🔥 Top {chosen_lang} {active_mood.title()} Mix",
                    f"https://www.youtube.com/results?search_query={latest_q}",
                    use_container_width=True,
                )

    live_mood_and_recommendation_panel(lang, actor, platform, cam_mode)


# =====================================================================
# PART 6: PAST USER HISTORY LOG DISPLAY
# =====================================================================
st.markdown("---")
st.subheader("📜 Your Past Actor, Language & Mood Recommendation History")

latest_hist = load_user_history().get("history", [])
if not latest_hist:
    st.caption(
        "No songs opened yet in this history log. Click the **Recommend & Open** button above and your favorite Actors, Languages, and Moods will be saved here automatically!"
    )
else:
    for idx, item in enumerate(latest_hist[:8]):
        h_mood = item.get("mood", "happy")
        h_emoji = MOOD_EMOJIS.get(h_mood, "🎵")
        h_actor = item.get("actor", "Any Artist")
        h_lang = item.get("language", "Hindi")
        h_time = item.get("timestamp", "")
        h_url = item.get("url", "https://www.youtube.com")
        h_plat = item.get("platform", "YouTube")

        hc1, hc2 = st.columns([4, 1])
        with hc1:
            st.markdown(
                f"{h_emoji} **{h_mood.title()} Mood** • 🌟 **{h_actor}** • 🌐 **{h_lang}** "
                f"<span style='color:gray;font-size:0.85rem;'>({h_plat} — {h_time})</span>",
                unsafe_allow_html=True,
            )
        with hc2:
            st.link_button(f"▶️ Play Again", h_url, use_container_width=True)
