import collections
import os
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

MODEL_PATH = "model.h5"
LABELS_PATH = "labels.npy"

model = load_model(MODEL_PATH, compile=False) if os.path.exists(MODEL_PATH) else None
labels = np.load(LABELS_PATH) if os.path.exists(LABELS_PATH) else None

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holis = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
drawing = mp.solutions.drawing_utils

recent_moods = collections.deque(maxlen=6)


def detect_mood_from_landmarks(res):
    if not res.face_landmarks:
        return "", 0.0, ""

    lm = res.face_landmarks.landmark
    has_lh = res.left_hand_landmarks is not None
    has_rh = res.right_hand_landmarks is not None

    nn_mood = "neutral"
    nn_conf = 0.0

    if model is not None and labels is not None:
        lst = []
        ref_x, ref_y = lm[1].x, lm[1].y
        for p in lm:
            lst.append(p.x - ref_x)
            lst.append(p.y - ref_y)

        if has_lh:
            lh_lm = res.left_hand_landmarks.landmark
            lh_x, lh_y = lh_lm[8].x, lh_lm[8].y
            for p in lh_lm:
                lst.append(p.x - lh_x)
                lst.append(p.y - lh_y)
        else:
            lst.extend([0.0] * 42)

        if has_rh:
            rh_lm = res.right_hand_landmarks.landmark
            rh_x, rh_y = rh_lm[8].x, rh_lm[8].y
            for p in rh_lm:
                lst.append(p.x - rh_x)
                lst.append(p.y - rh_y)
        else:
            lst.extend([0.0] * 42)

        try:
            features = np.array(lst, dtype=np.float32).reshape(1, -1)
            preds = model.predict(features, verbose=0)[0]
            idx = int(np.argmax(preds))
            nn_conf = float(preds[idx]) * 100.0
            if idx < len(labels):
                nn_mood = str(labels[idx])
        except Exception:
            pass

    face_w = max(abs(lm[454].x - lm[234].x), 1e-4)
    face_h = max(abs(lm[152].y - lm[10].y), 1e-4)

    lip_center_y = (lm[13].y + lm[14].y) / 2.0
    corner_avg_y = (lm[61].y + lm[291].y) / 2.0
    smile_lift = (lip_center_y - corner_avg_y) / face_h
    mouth_w_ratio = abs(lm[291].x - lm[61].x) / face_w
    mouth_open_ratio = abs(lm[14].y - lm[13].y) / face_h
    brow_lift_ratio = ((lm[159].y - lm[105].y) + (lm[386].y - lm[334].y)) / (2.0 * face_h)
    head_tilt = abs(lm[454].y - lm[234].y) / face_w

    if mouth_open_ratio > 0.125:
        return "shocked", max(nn_conf, 95.0), "Wide open mouth"

    if (mouth_w_ratio > 0.422 and smile_lift > 0.012) or (smile_lift > 0.018 and mouth_open_ratio < 0.11):
        if head_tilt > 0.115 and not has_lh and not has_rh:
            return "romantic", 92.0, "Warm smile + head tilt"
        return "happy", max(nn_conf, 94.0), "Smiling"

    if nn_conf >= 65.0:
        if not has_lh and not has_rh:
            if brow_lift_ratio < 0.095 and smile_lift < -0.005 and mouth_open_ratio < 0.03:
                return "angry", 88.0, "Furrowed brows + tense mouth"
            if smile_lift < -0.004 and nn_mood not in ("sad", "not_ok"):
                return "sad", 89.0, "Downturned mouth corners"
        return nn_mood, nn_conf, f"AI Model ({nn_mood})"

    if has_lh and has_rh:
        return "energy", 90.0, "Both hands active"
    if smile_lift > 0.012:
        return "happy", 85.0, "Gentle smile"
    if smile_lift < 0.002 or brow_lift_ratio < 0.112:
        return "sad", 84.0, "Sad / serious expression"
    if mouth_open_ratio > 0.065:
        return "surprised", 85.0, "Surprised open mouth"

    return nn_mood or "neutral", max(nn_conf, 75.0), "Detected from landmarks"


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access webcam.")
    exit(1)

print("[OK] Starting live emotion detection... Press ESC or 'q' to quit.")

while True:
    ret, frm = cap.read()
    if not ret:
        continue

    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    mood, conf, detail = detect_mood_from_landmarks(res)
    if mood:
        recent_moods.append(mood)
        smooth_mood = collections.Counter(recent_moods).most_common(1)[0][0]
        cv2.putText(
            frm,
            f"MOOD: {smooth_mood.upper()} ({conf:.0f}%)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if res.face_landmarks:
        drawing.draw_landmarks(frm, res.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if res.left_hand_landmarks:
        drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Mood Detection (Press ESC or Q to Exit)", frm)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
