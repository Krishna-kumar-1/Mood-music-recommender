# 🎵 Mood-Based Music Recommendation System

Real-time facial-emotion and hand-gesture detection that recommends personalized music based on how you're feeling — built with **Streamlit**, **MediaPipe Holistic**, **OpenCV**, and **TensorFlow/Keras**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-red?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

![Demo Banner](assets/demo_banner.png)

---

## 📌 Overview

The **Mood-Based Music Recommendation System** captures facial expressions and hand gestures live from your webcam using **MediaPipe Holistic**, then classifies your emotional state (`happy`, `sad`, `energy`, `shocked`, `surprised`, etc.) using a **deep neural network**. 

Once your emotion is detected, the built-in **Streamlit web application** allows you to choose your preferred language and favorite artist, instantly generating curated music recommendations and launching your playlist on YouTube!

![Emotion Landmarks](assets/emotion_diagram.png)

---

## ✨ Features

- **Interactive Web App**: Modern, responsive UI powered by Streamlit and Streamlit-WebRTC for browser-based live video processing.
- **Customizable Music Recommendations**: Filter song recommendations by language (English, Hindi, Spanish, Punjabi, Tamil, etc.) and artist/singer.
- **Real-Time Facial & Hand Tracking**: Tracks 468 facial landmarks and 42 hand landmarks simultaneously without external sensors.
- **Position-Invariant Landmark Normalization**: Distances normalized relative to facial and hand reference points for robust detection regardless of distance from the camera.
- **Custom Training Pipeline**: Easily record new mood/gesture data and retrain the classifier in seconds.
- **Dual Mode**: Run the interactive web app (`app.py`) or lightweight standalone camera inference (`inference.py`).

---

## 📂 Project Structure

```
Mood-music-recommender/
│
├── app.py                  # 🚀 Main Streamlit web application (Live camera & music recommendations)
├── inference.py            # 👁️ Standalone OpenCV live emotion detection
├── train.py                # 🧠 Train the neural network on landmark datasets
├── data_collection.py      # 📹 Record custom emotion samples from webcam
├── model.h5                # 📦 Pretrained Keras emotion classifier
├── labels.npy              # 🏷️ Class labels for the trained model
│
├── data/                   # 📁 Recorded landmark datasets (.npy files)
│   ├── happy.npy
│   ├── sad.npy
│   ├── energy.npy
│   ├── not_ok.npy
│   ├── shocked.npy
│   └── surprised.npy
│
├── assets/                 # 🖼️ Diagrams and banner images
│   ├── demo_banner.png
│   └── emotion_diagram.png
│
├── requirements.txt        # 📋 Python dependencies
├── .gitignore              # 🚫 Git ignore rules (filters large media and caches)
└── LICENSE                 # 📄 MIT License
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Krishna-kumar-1/Mood-music-recommender.git
cd Mood-music-recommender
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Music Recommender Web App
```bash
streamlit run app.py
```
This will open the web app in your browser at `http://localhost:8501`. 
1. Click **START** on the webcam feed.
2. Select your preferred language and optional singer.
3. Click **🎵 Recommend Me Songs!** to open your mood playlist!

---

## 💻 Standalone Live Detection

To run emotion detection directly in an OpenCV window without the web interface:

```bash
python inference.py
```
*Press **ESC** or **q** to close the window.*

---

## 🏋️‍♂️ Train on Custom Emotions

Want to train the model to recognize your own gestures or new emotions?

### Step 1: Collect Landmark Data
```bash
python data_collection.py
```
Enter an emotion name (e.g. `happy`, `sad`, `focus`) and pose in front of the camera. The script automatically records 100 frames of normalized landmarks into `data/<emotion>.npy`.

### Step 2: Train the Model
```bash
python train.py
```
The script will load all datasets in `data/`, train a multi-class neural network, and update `model.h5` and `labels.npy`.

---

## 🔬 How It Works

1. **Landmark Extraction**: MediaPipe Holistic extracts 468 3D facial coordinates and 21 3D landmarks per hand.
2. **Feature Normalization**:
   - Facial coordinates are normalized relative to the nose tip (landmark index `1`).
   - Hand coordinates are normalized relative to the index fingertip (landmark index `8`).
   - Missing hands are zero-padded to maintain a consistent 1,020-element feature vector.
3. **Classification**: A deep neural network (Dense 512 → Dropout → Dense 256 → Dropout → Softmax) predicts probabilities across all emotion classes.
4. **Music Mapping**: The predicted mood is dynamically mapped to a YouTube search query incorporating user-selected language and artist preferences.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web UI & Streaming | Streamlit, Streamlit-WebRTC, PyAV |
| Vision & Tracking | MediaPipe Holistic, OpenCV |
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
