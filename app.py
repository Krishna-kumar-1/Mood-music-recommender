import os
import subprocess
import sys
import threading
import urllib.parse
import webbrowser

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

# --- Page Configuration ---
st.set_page_config(
    page_title="Mood Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
<style>
    .hero-banner {
        text-align: center;
        padding: 1.8rem 1rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a11cb 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .hero-banner p {
        margin: 0.5rem 0 0 0;
        opacity: 0.92;
        font-size: 1.05rem;
    }
    .mood-badge {
        padding: 1.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #fff;
        text-align: center;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(17, 153, 142, 0.25);
    }
    .stButton>button {
        border-radius: 10px;
        height: 3.1em;
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
    <p>AI-powered facial &amp; gesture emotion recognition that curates songs for how you feel right now</p>
</div>
""",
    unsafe_allow_html=True,
)


# --- Load Model & Labels ---
@st.cache_resource
def load_emotion_model():
    model_path = "model.h5"
    labels_path = "labels.npy"
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None
    loaded_model = load_model(model_path, compile=False)
    loaded_labels = np.load(labels_path)
    return loaded_model, loaded_labels


model, labels = load_emotion_model()

if model is None or labels is None:
    st.error(
        "Model files (`model.h5` or `labels.npy`) were not found. Click **Retrain Model** in the sidebar or run `python train.py`."
    )


# --- Shared Thread-Safe State ---
class EmotionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_emotion = ""
        self.confidence = 0.0

    def update(self, emotion: str, confidence: float):
        with self.lock:
            self.current_emotion = emotion
            self.confidence = confidence

    def get(self):
        with self.lock:
            return self.current_emotion, self.confidence


if "emotion_state" not in st.session_state:
    st.session_state["emotion_state"] = EmotionState()

emotion_state: EmotionState = st.session_state["emotion_state"]

# MediaPipe initialization
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmark_features(results) -> list:
    """Extract 1,020 normalized landmark coordinates from face and both hands."""
    lst = []
    if not results.face_landmarks:
        return lst

    ref_x = results.face_landmarks.landmark[1].x
    ref_y = results.face_landmarks.landmark[1].y
    for lm in results.face_landmarks.landmark:
        lst.append(lm.x - ref_x)
        lst.append(lm.y - ref_y)

    if results.left_hand_landmarks:
        lh_x = results.left_hand_landmarks.landmark[8].x
        lh_y = results.left_hand_landmarks.landmark[8].y
        for lm in results.left_hand_landmarks.landmark:
            lst.append(lm.x - lh_x)
            lst.append(lm.y - lh_y)
    else:
        lst.extend([0.0] * 42)

    if results.right_hand_landmarks:
        rh_x = results.right_hand_landmarks.landmark[8].x
        rh_y = results.right_hand_landmarks.landmark[8].y
        for lm in results.right_hand_landmarks.landmark:
            lst.append(lm.x - rh_x)
            lst.append(lm.y - rh_y)
    else:
        lst.extend([0.0] * 42)

    return lst


def draw_styled_landmarks(frame, results):
    """Draw face and hand landmarks onto the BGR video frame."""
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 128), thickness=1
            ),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 80, 0), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 255), thickness=2
            ),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 120, 255), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 0), thickness=2
            ),
        )


def predict_from_bgr_frame(frm):
    """Run MediaPipe Holistic + Keras inference on a single BGR frame."""
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(
        static_image_mode=True, min_detection_confidence=0.5
    ) as holistic:
        results = holistic.process(rgb)

    annotated = frm.copy()
    draw_styled_landmarks(annotated, results)

    if not results.face_landmarks or model is None or labels is None:
        return annotated, "", 0.0

    lst = extract_landmark_features(results)
    features = np.array(lst).reshape(1, -1)
    preds = model.predict(features, verbose=0)
    idx = int(np.argmax(preds))
    conf = float(preds[0][idx]) * 100.0
    mood = str(labels[idx]) if idx < len(labels) else ""

    if mood:
        cv2.putText(
            annotated,
            f"Mood: {mood.upper()} ({conf:.1f}%)",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated, mood, conf


RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class EmotionVideoProcessor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        detected_emotion = ""
        confidence = 0.0

        if results.face_landmarks and model is not None and labels is not None:
            lst = extract_landmark_features(results)
            try:
                features = np.array(lst).reshape(1, -1)
                preds = model.predict(features, verbose=0)
                idx = int(np.argmax(preds))
                confidence = float(preds[0][idx]) * 100.0
                if idx < len(labels):
                    detected_emotion = str(labels[idx])
                    emotion_state.update(detected_emotion, confidence)
            except Exception:
                pass

        draw_styled_landmarks(frm, results)

        if detected_emotion:
            cv2.putText(
                frm,
                f"Mood: {detected_emotion.upper()} ({confidence:.0f}%)",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ System & Model Info")
    if labels is not None:
        st.success(f"Loaded **{len(labels)}** emotion classes:")
        st.write(", ".join([f"`{lbl}`" for lbl in labels]))
    else:
        st.warning("No model loaded.")

    st.markdown("---")
    st.subheader("🔄 Model Management")
    if st.button("🧠 Retrain Model on data/", use_container_width=True):
        with st.spinner("Training neural network on datasets in data/..."):
            res = subprocess.run(
                [sys.executable, "train.py"], capture_output=True, text=True
            )
            if res.returncode == 0:
                st.cache_resource.clear()
                st.success("Model retrained successfully! Reloading...")
                st.rerun()
            else:
                st.error(f"Training failed:\n{res.stderr}")

    st.markdown("---")
    st.caption(
        "Tip: Switch between **Snapshot Camera** (fastest & works everywhere) and **Live WebRTC Stream** anytime."
    )


# --- Main Layout ---
col_cam, col_music = st.columns([3, 2], gap="large")

with col_cam:
    st.subheader("📹 Step 1: Detect Your Mood")
    cam_mode = st.radio(
        "Choose Detection Method:",
        [
            "📸 Quick Camera Snapshot (Recommended - Works Everywhere)",
            "🎥 Live WebRTC Video Stream (Real-Time Continuous)",
            "🖐️ Manual Mood Selection",
        ],
        index=0,
    )

    if cam_mode.startswith("📸"):
        img_file = st.camera_input("Pose & take a snapshot to detect your emotion:")
        if img_file is not None:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated_img, snap_mood, snap_conf = predict_from_bgr_frame(bgr_img)
            st.image(
                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                caption="Analyzed Landmarks & Emotion",
                use_container_width=True,
            )
            if snap_mood:
                emotion_state.update(snap_mood, snap_conf)
            else:
                st.warning(
                    "Could not detect face landmarks clearly. Please make sure your face is well-lit and centered."
                )

    elif cam_mode.startswith("🎥"):
        st.caption(
            "Click **START** below to stream live video and detect your expressions continuously."
        )
        webrtc_streamer(
            key="emotion-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=EmotionVideoProcessor,
            async_processing=True,
        )
        if st.button("🔄 Refresh Detected Mood from Live Stream"):
            st.rerun()

    else:
        st.info("Use the mood selector on the right to pick your mood directly.")


with col_music:
    st.subheader("🎧 Step 2: Music Preferences")

    popular_languages = [
        "Hindi",
        "English",
        "Punjabi",
        "Tamil",
        "Telugu",
        "Spanish",
        "Korean",
        "Japanese",
        "Other",
    ]
    selected_lang = st.selectbox("Preferred Song Language", popular_languages, index=0)
    if selected_lang == "Other":
        lang = st.text_input("Enter custom language", value="English").strip()
    else:
        lang = selected_lang

    singer = st.text_input(
        "Favorite Singer / Artist (Optional)",
        placeholder="e.g. Arijit Singh, The Weeknd, Ed Sheeran, Taylor Swift",
    )

    platform = st.selectbox(
        "Preferred Music Platform",
        ["YouTube", "YouTube Music", "Spotify"],
        index=0,
    )

    st.markdown("---")
    st.subheader("🎯 Active Mood")

    detected_mood, detected_conf = emotion_state.get()
    available_moods = (
        list(labels)
        if labels is not None
        else ["happy", "sad", "energy", "shocked", "surprised", "not_ok"]
    )

    if detected_mood and not cam_mode.startswith("🖐️"):
        st.markdown(
            f'<div class="mood-badge">Detected Mood: {detected_mood.upper()} ({detected_conf:.0f}%)</div>',
            unsafe_allow_html=True,
        )
        active_mood = detected_mood
    else:
        active_mood = st.selectbox(
            "Select or Override Your Current Mood:",
            available_moods,
            index=0,
        )

    # Map internal class names like 'not_ok' to natural music search keywords
    mood_search_map = {
        "happy": "happy upbeat feel good",
        "sad": "sad emotional soothing",
        "energy": "energetic workout hype",
        "not_ok": "calm relaxing healing",
        "shocked": "energeticelectronic",
        "surprised": "upbeat pop party",
    }
    search_mood_keyword = mood_search_map.get(active_mood.lower(), active_mood)

    query_parts = [lang, search_mood_keyword, "songs"]
    if singer.strip():
        query_parts.append(singer.strip())
    search_query = " ".join(query_parts)
    encoded_query = urllib.parse.quote_plus(search_query)

    yt_url = f"https://www.youtube.com/results?search_query={encoded_query}"
    yt_music_url = f"https://music.youtube.com/search?q={encoded_query}"
    spotify_url = f"https://open.spotify.com/search/{urllib.parse.quote(search_query)}"

    platform_urls = {
        "YouTube": yt_url,
        "YouTube Music": yt_music_url,
        "Spotify": spotify_url,
    }
    primary_url = platform_urls[platform]

    st.markdown("### 🚀 Step 3: Get Recommendations")

    if st.button("🎵 Recommend & Open Songs Now!", type="primary", use_container_width=True):
        st.success(
            f"Opening **{active_mood.upper()}** songs in **{lang}** on **{platform}**!"
        )
        try:
            webbrowser.open_new_tab(primary_url)
        except Exception:
            pass

    st.markdown("#### 🔗 Direct Quick-Launch Links")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("▶️ YouTube", yt_url, use_container_width=True)
    with c2:
        st.link_button("🎧 YT Music", yt_music_url, use_container_width=True)
    with c3:
        st.link_button("🟢 Spotify", spotify_url, use_container_width=True)

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: gray; font-size: 0.88rem;">
    Mood-Based Music Recommendation System • Built with MediaPipe Holistic, TensorFlow/Keras &amp; Streamlit
</div>
""",
    unsafe_allow_html=True,
)
