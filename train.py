import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout
from keras.models import Model

DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist. Run data_collection.py first.")

x_data = []
y_labels = []
label_names = []
label_to_idx = {}

print("Scanning datasets in data/...")
# Sort file names for consistent label index ordering
npy_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npy")])

for file_name in npy_files:
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        arr = np.load(file_path)
    except Exception as e:
        print(f"Skipping {file_name} (failed to load: {e})")
        continue

    # Skip files that are not landmark feature datasets
    if arr.ndim != 2 or arr.shape[1] < 100:
        print(f"Skipping non-feature file: {file_name} (shape: {arr.shape})")
        continue

    emotion_label = file_name.rsplit(".", 1)[0]
    if emotion_label not in label_to_idx:
        label_to_idx[emotion_label] = len(label_names)
        label_names.append(emotion_label)

    print(f"Loaded {emotion_label}: {arr.shape[0]} samples (features: {arr.shape[1]})")
    x_data.append(arr)
    y_labels.extend([label_to_idx[emotion_label]] * arr.shape[0])

if not x_data:
    raise ValueError("No valid landmark training data found in data/. Please collect data first.")

X = np.vstack(x_data)
Y = np.array(y_labels, dtype="int32")
Y_cat = to_categorical(Y, num_classes=len(label_names))

print(f"\nTotal dataset shape: X={X.shape}, Y={Y_cat.shape}")
print(f"Classes ({len(label_names)}): {label_names}")

# Shuffle dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_shuffled = X[indices]
Y_shuffled = Y_cat[indices]

# Define Model Architecture
input_shape = (X.shape[1],)
ip = Input(shape=input_shape)
m = Dense(512, activation="relu")(ip)
m = Dropout(0.2)(m)
m = Dense(256, activation="relu")(m)
m = Dropout(0.2)(m)
op = Dense(Y_cat.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

print("\nTraining neural network...")
model.fit(
    X_shuffled,
    Y_shuffled,
    epochs=50,
    batch_size=32,
    validation_split=0.15,
    verbose=1
)

# Save trained artifacts
model.save("model.h5")
np.save("labels.npy", np.array(label_names))
print("\n[SUCCESS] Successfully trained and saved model.h5 and labels.npy!")
