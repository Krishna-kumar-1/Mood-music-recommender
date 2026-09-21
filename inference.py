import os
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

MODEL_PATH = "model.h5"
LABELS_PATH = "labels.npy"

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print("[ERROR] Trained model files not found.")
    print("Please run 'python train.py' to generate model.h5 and labels.npy.")
    exit(1)

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)
print(f"[OK] Loaded model with classes: {list(labels)}")

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holis = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access webcam.")
    exit(1)

print("Starting live emotion detection... Press ESC or 'q' to quit.")

while True:
    ret, frm = cap.read()
    if not ret:
        continue

    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    lst = []

    if res.face_landmarks:
        # Normalize relative to nose landmark [1]
        ref_x = res.face_landmarks.landmark[1].x
        ref_y = res.face_landmarks.landmark[1].y
        for i in res.face_landmarks.landmark:
            lst.append(i.x - ref_x)
            lst.append(i.y - ref_y)

        if res.left_hand_landmarks:
            lh_ref_x = res.left_hand_landmarks.landmark[8].x
            lh_ref_y = res.left_hand_landmarks.landmark[8].y
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - lh_ref_x)
                lst.append(i.y - lh_ref_y)
        else:
            lst.extend([0.0] * 42)

        if res.right_hand_landmarks:
            rh_ref_x = res.right_hand_landmarks.landmark[8].x
            rh_ref_y = res.right_hand_landmarks.landmark[8].y
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - rh_ref_x)
                lst.append(i.y - rh_ref_y)
        else:
            lst.extend([0.0] * 42)

        features = np.array(lst).reshape(1, -1)
        predictions = model.predict(features, verbose=0)
        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[0][pred_idx]) * 100

        if pred_idx < len(labels):
            pred_text = f"{labels[pred_idx].upper()} ({confidence:.1f}%)"
            cv2.putText(
                frm,
                pred_text,
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    # Draw landmarks
    if res.face_landmarks:
        drawing.draw_landmarks(frm, res.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if res.left_hand_landmarks:
        drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Emotion Detection (Inference)", frm)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
