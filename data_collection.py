import os
import cv2
import numpy as np
import mediapipe as mp

os.makedirs("data", exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam. Please check your camera connection.")
    exit(1)

name = input("Enter the name of the emotion/mood data (e.g., happy, sad, energy, surprised): ").strip()
if not name:
    print("❌ Error: Mood name cannot be empty.")
    exit(1)

# Sanitize file name
name = name.replace(" ", "_").lower()

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holis = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing = mp.solutions.drawing_utils

X = []
data_size = 0
TARGET_SIZE = 100

print(f"\nRecording data for '{name}'. Please pose in front of the camera...")
print("Press ESC or 'q' to stop early.")

while True:
    ret, frm = cap.read()
    if not ret:
        print("Warning: Failed to capture frame from webcam.")
        continue

    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    lst = []

    if res.face_landmarks:
        # Face relative to nose bridge / center
        ref_x = res.face_landmarks.landmark[1].x
        ref_y = res.face_landmarks.landmark[1].y
        for i in res.face_landmarks.landmark:
            lst.append(i.x - ref_x)
            lst.append(i.y - ref_y)

        # Left hand
        if res.left_hand_landmarks:
            lh_ref_x = res.left_hand_landmarks.landmark[8].x
            lh_ref_y = res.left_hand_landmarks.landmark[8].y
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - lh_ref_x)
                lst.append(i.y - lh_ref_y)
        else:
            lst.extend([0.0] * 42)

        # Right hand
        if res.right_hand_landmarks:
            rh_ref_x = res.right_hand_landmarks.landmark[8].x
            rh_ref_y = res.right_hand_landmarks.landmark[8].y
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - rh_ref_x)
                lst.append(i.y - rh_ref_y)
        else:
            lst.extend([0.0] * 42)

        X.append(lst)
        data_size += 1

    # Visual feedback
    if res.face_landmarks:
        drawing.draw_landmarks(frm, res.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if res.left_hand_landmarks:
        drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    progress_text = f"Samples: {data_size}/{TARGET_SIZE} ({name})"
    cv2.putText(frm, progress_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Data Collection Window", frm)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or data_size >= TARGET_SIZE:
        break

cap.release()
cv2.destroyAllWindows()

if data_size > 0:
    out_path = os.path.join("data", f"{name}.npy")
    np.save(out_path, np.array(X))
    print(f"\n✅ Saved {data_size} frames to '{out_path}' (shape: {np.array(X).shape})")
else:
    print("\n⚠️ No frames captured. Please ensure your face is clearly visible to the camera.")
