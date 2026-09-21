import os
import threading
import webbrowser
import urllib.parse
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import mediapipe as mp
from keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Mood Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for modern UI styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 5px solid #4b6cb7;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎵 Mood-Based Music Recommendation System</h1>
    <p>Detect your emotional state in real-time and discover songs that match your mood!</p>
</div>
""", unsafe_allow_html=True)

# --- Load Model & Labels ---
@st.cache_resource
def load_emotion_model():
    model_path = "model.h5"
    labels_path = "labels.npy"
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        st.error("⚠️ Model files (`model.h5` or `labels.npy`) not found. Please train the model first using `python train.py`.")
        return None, None
    model = load_model(model_path)
    labels = np.load(labels_path)
    return model, labels

model, labels = load_emotion_model()

# --- Shared Thread-Safe State for WebRTC ---
class EmotionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_emotion = ""

    def set_emotion(self, emotion: str):
        with self.lock:
            self.current_emotion = emotion

    def get_emotion(self) -> str:
        with self.lock:
            return self.current_emotion

if "emotion_state" not in st.session_state:
    st.session_state["emotion_state"] = EmotionState()

emotion_state: EmotionState = st.session_state["emotion_state"]

# WebRTC STUN server configuration
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class EmotionVideoProcessor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        detected_emotion = ""
        lst = []

        if results.face_landmarks and model is not None and labels is not None:
            # Face landmarks relative to reference point
            ref_x = results.face_landmarks.landmark[1].x
            ref_y = results.face_landmarks.landmark[1].y
            for lm in results.face_landmarks.landmark:
                lst.append(lm.x - ref_x)
                lst.append(lm.y - ref_y)

            # Left hand landmarks
            if results.left_hand_landmarks:
                lh_ref_x = results.left_hand_landmarks.landmark[8].x
                lh_ref_y = results.left_hand_landmarks.landmark[8].y
                for lm in results.left_hand_landmarks.landmark:
                    lst.append(lm.x - lh_ref_x)
                    lst.append(lm.y - lh_ref_y)
            else:
                lst.extend([0.0] * 42)

            # Right hand landmarks
            if results.right_hand_landmarks:
                rh_ref_x = results.right_hand_landmarks.landmark[8].x
                rh_ref_y = results.right_hand_landmarks.landmark[8].y
                for lm in results.right_hand_landmarks.landmark:
                    lst.append(lm.x - rh_ref_x)
                    lst.append(lm.y - rh_ref_y)
            else:
                lst.extend([0.0] * 42)

            try:
                features = np.array(lst).reshape(1, -1)
                predictions = model.predict(features, verbose=0)
                pred_idx = int(np.argmax(predictions))
                if pred_idx < len(labels):
                    detected_emotion = str(labels[pred_idx])
                    emotion_state.set_emotion(detected_emotion)
            except Exception as err:
                pass

        # Draw overlays on video
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frm,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frm,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frm,
                results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
            )

        if detected_emotion:
            cv2.putText(
                frm,
                f"Mood: {detected_emotion.upper()}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# --- UI Layout ---
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("📹 Real-Time Mood Detection")
    st.caption("Start the webcam feed to allow the AI to detect your emotional state.")

    webrtc_ctx = webrtc_streamer(
        key="emotion-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=EmotionVideoProcessor,
        async_processing=True
    )

with col2:
    st.subheader("🎧 Music Preferences")
    
    # Language input
    popular_languages = ["English", "Hindi", "Spanish", "Punjabi", "Tamil", "Korean", "Japanese", "Other"]
    selected_lang = st.selectbox("Preferred Language", popular_languages, index=0)
    if selected_lang == "Other":
        custom_lang = st.text_input("Enter your preferred language", value="English")
        lang = custom_lang.strip()
    else:
        lang = selected_lang

    # Singer / Artist input
    singer = st.text_input("Favorite Singer / Artist (Optional)", placeholder="e.g. Arijit Singh, Ed Sheeran, Taylor Swift")

    st.markdown("---")
    
    current_mood = emotion_state.get_emotion()
    
    # Manual emotion fallback if user doesn't have camera or wants manual selection
    st.subheader("🎯 Detected Mood")
    if current_mood:
        st.success(f"Current Detected Mood: **{current_mood.upper()}** 🎉")
        active_mood = current_mood
    else:
        st.info("Looking for webcam input... You can also select a mood manually below:")
        preset_emotions = ["happy", "sad", "energy", "shocked", "surprised", "neutral"]
        active_mood = st.selectbox("Manual Mood Selection", preset_emotions)

    st.markdown("### 🚀 Get Recommendations")
    
    if st.button("🎵 Recommend Me Songs!", use_container_width=True):
        if not active_mood:
            st.warning("Please look at the camera to detect your mood, or select one manually above.")
        else:
            query_parts = [lang, active_mood, "songs"]
            if singer.strip():
                query_parts.append(singer.strip())
            
            search_query = " ".join(query_parts)
            yt_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(search_query)}"
            
            st.success(f"Playing recommendations for **{active_mood.title()}** mood in **{lang}**!")
            
            # Direct link button
            st.link_button("▶️ Open YouTube Music Recommendations", yt_url, use_container_width=True)
            
            # Try to open in local browser if running on desktop
            try:
                webbrowser.open_new_tab(yt_url)
            except Exception:
                pass

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.85rem;">
    Powered by MediaPipe Holistic, TensorFlow/Keras, and Streamlit • Created by Krish
</div>
""", unsafe_allow_html=True)
