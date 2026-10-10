# 🎵 Mood-Based Music Recommendation System

Real-time facial-emotion and hand-gesture detection that recommends personalized music based on how you're feeling — built with **Streamlit**, **MediaPipe Holistic**, **OpenCV**, and **TensorFlow/Keras**.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-red?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

![Demo Banner](assets/demo_banner.png)

---

## 🚀 One-Click App Start (Zero Configuration!)

You do **not** need to manually set up environments or install packages. Just double-click the launcher icon:

- **Windows**: Double-click **`Mood Music Recommender`** (shortcut icon) or **`Start_App.bat`**
- **macOS / Linux**: Run **`./setup.sh`**
- **Any Platform (Python)**: Run **`python run.py`**

### What the Smart Launcher Does Automatically:
1. **Checks Requirements**: Checks whether your Python environment and all required libraries (`streamlit`, `mediapipe`, `tensorflow`, `opencv`, `streamlit-webrtc`, `numpy`, `av`) are already installed.
2. **Auto-Installs if Missing**: If you haven't installed the requirements yet, it automatically creates an isolated environment and installs all basic and deep-learning dependencies for you.
3. **Instant Launch & Open**: If your requirements are already satisfied, it skips installation completely and **immediately launches and opens the web app** in your browser!

---

## 📌 Overview

The **Mood-Based Music Recommendation System** captures facial expressions and hand gestures from your webcam using **MediaPipe Holistic**, then classifies your emotional state (`happy`, `sad`, `energy`, `not_ok`, `shocked`, `surprised`) using a **deep neural network**.

Once your mood is detected, the interactive **Streamlit web application** lets you choose your preferred language, favorite artist, and platform (**YouTube**, **YouTube Music**, or **Spotify**), instantly curating and opening songs that match your current mood!

![Emotion Landmarks](assets/emotion_diagram.png)

---

## ✨ Key Features

- **🖱️ Single-Icon Smart Start**: One click checks/installs missing dependencies or immediately launches the app if everything is ready.
- **📸 3 Flexible Detection Modes**:
  1. **Quick Camera Snapshot**: Works across all browsers and networks with zero firewall/WebRTC setup.
  2. **Live WebRTC Video Stream**: Continuous real-time 30 FPS facial and hand landmark tracking.
  3. **Manual Mood Selector**: Pick or override your mood anytime even without a webcam.
- **🎧 Multi-Platform Music Recommendations**: Open curated playlists on **YouTube**, **YouTube Music**, or **Spotify** filtered by language (Hindi, English, Punjabi, Tamil, Telugu, Spanish, Korean, Japanese, etc.) and singer.
- **🧠 1-Click Model Retraining**: Retrain the neural network directly from the web app sidebar or via `python run.py --mode train`.

---

## 📂 Project Structure

```
Mood-music-recommender/
│
├── Start_App.bat                 # 🖱️ One-Click Windows App Launcher (Auto-installs & opens app)
├── Mood Music Recommender.lnk    # 🎵 Single App Start Icon with custom badge
├── run.py                        # 🧠 Universal Smart Launcher & Dependency Manager
├── app.py                        # 🌐 Main Streamlit Web Application (Snapshot + Live WebRTC + Music UI)
├── inference.py                  # 👁️ Standalone OpenCV Window Emotion Detector
├── train.py                      # 🏋️ Neural Network Training Pipeline
├── data_collection.py            # 📹 Webcam Landmark Dataset Recorder
├── create_shortcut.py            # 🎨 Generates custom app_icon.ico and Desktop shortcut
├── model.h5                      # 📦 Trained Keras Emotion Classifier
├── labels.npy                    # 🏷️ Synchronized Emotion Class Labels
│
├── data/                         # 📁 Recorded Landmark Datasets (.npy files)
│   ├── energy.npy
│   ├── happy.npy
│   ├── not_ok.npy
│   ├── sad.npy
│   ├── shocked.npy
│   └── surprised.npy
│
├── assets/                       # 🖼️ Icons, Banners & Diagrams
│   ├── app_icon.ico
│   ├── demo_banner.png
│   └── emotion_diagram.png
│
├── requirements.txt              # 📋 Pinned Compatible Dependencies
├── setup.bat / setup.sh          # ⚡ Cross-Platform Setup Scripts
├── .gitignore                    # 🚫 Git Ignore Configuration
└── LICENSE                       # 📄 MIT License
```

---

## 💻 Advanced / CLI Usage

You can also use `run.py` to launch specific modes directly from the terminal:

```bash
# Launch the interactive Web Application (default)
python run.py

# Launch standalone OpenCV webcam window inference
python run.py --mode camera

# Record new emotion landmark data from your webcam
python run.py --mode collect

# Retrain the neural network on all datasets in data/
python run.py --mode train
```

---

## 🔬 How It Works

1. **Landmark Extraction**: MediaPipe Holistic extracts 468 facial landmarks and 21 landmarks per hand.
2. **Position-Invariant Normalization**:
   - Facial coordinates are normalized relative to the nose bridge (landmark index `1`).
   - Hand coordinates are normalized relative to the index fingertip (landmark index `8`).
   - Missing hands are zero-padded to maintain a 1,020-dimensional feature vector.
3. **Deep Neural Network**: A multi-layer dense network (`Dense 512` → `Dropout 0.2` → `Dense 256` → `Dropout 0.2` → `Softmax`) classifies the 1,020 features into emotion categories.
4. **Smart Music Query Mapping**: The detected emotion, chosen language, and optional artist are mapped to natural mood keywords and launched on YouTube, YouTube Music, or Spotify.

---

## 📄 License

Licensed under the [MIT License](LICENSE).
