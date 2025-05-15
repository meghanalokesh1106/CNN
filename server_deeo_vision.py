import cv2
import numpy as np
import base64
import subprocess
import binascii
from collections import deque, Counter
from tensorflow.keras.models import load_model
import sys

# === Load model ===
model = load_model("best_model_CNN_+_SE")

# === Class names ===
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# === Choose camera source ===
source_type = "nao"  # "webcam" or "nao"

# === Initialize webcam if selected ===
if source_type == "webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
else:
    cap = None  # NAO camera uses subprocess

# === Frame retrieval function ===
def get_frame(source="webcam"):
    if source == "webcam":
        ret, frame = cap.read()
        return frame if ret else None
    elif source == "nao":
        try:
            output = subprocess.check_output(
                [".venv/Scripts/python.exe", "get_camera_deep_vision.py"],
                stderr=subprocess.STDOUT
            )
            encoded = output.strip().splitlines()[-1]
            img_data = base64.b64decode(encoded)
            npimg = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            return frame
        except subprocess.CalledProcessError as e:
            print("Subprocess error:\n", e.output.decode("utf-8"))
        except binascii.Error as e:
            print("Base64 decode error:", e)
        except Exception as e:
            print("Unexpected error:", e)
    return None

# === Sliding window for predictions ===
prediction_window = deque(maxlen=5)

# === Main loop ===
frame_count = 0  # Counter to skip frames

recall = None
while True:
    frame = get_frame(source=source_type)
    if frame is None:
        print("Error: Could not retrieve frame.")
        break

    # frame_count += 1
    # if frame_count % 2 != 0:
    #     # Skip this frame
    #     continue

    # Resize to 32x32 and normalize
    resized_frame = cv2.resize(frame, (32, 32))
    normalized_frame = resized_frame.astype("float32") / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)

    # Predict
    pred_probs = model.predict(input_tensor)
    predicted_class = np.argmax(pred_probs, axis=1)[0]
    label = class_names[predicted_class]

    # Update prediction history
    prediction_window.append(label)

    # Determine if any class meets threshold
    confirmed_label = None
    if len(prediction_window) == 5:
        counts = Counter(prediction_window)
        for lbl, cnt in counts.items():
            if cnt >= 4:
                confirmed_label = lbl
                break

    # Display result
    display_text = f"Prediction: {confirmed_label if confirmed_label else 'Analyzing...'}"
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    if confirmed_label and recall != confirmed_label:
        recall = confirmed_label
        try:
            subprocess.Popen([
                "import cv2
import numpy as np
import base64
import subprocess
import binascii
from collections import deque, Counter
from tensorflow.keras.models import load_model
import sys

# === Load model ===
model = load_model("best_model_CNN_+_SE")

# === Class names ===
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# === Choose camera source ===
source_type = "nao"  # "webcam" or "nao"

# === Initialize webcam if selected ===
if source_type == "webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
else:
    cap = None  # NAO camera uses subprocess

# === Frame retrieval function ===
def get_frame(source="webcam"):
    if source == "webcam":
        ret, frame = cap.read()
        return frame if ret else None
    elif source == "nao":
        try:
            output = subprocess.check_output(
                [".venv/Scripts/python.exe", "camera.py"],
                stderr=subprocess.STDOUT
            )
            encoded = output.strip().splitlines()[-1]
            img_data = base64.b64decode(encoded)
            npimg = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            return frame
        except subprocess.CalledProcessError as e:
            print("Subprocess error:\n", e.output.decode("utf-8"))
        except binascii.Error as e:
            print("Base64 decode error:", e)
        except Exception as e:
            print("Unexpected error:", e)
    return None

# === Sliding window for predictions ===
prediction_window = deque(maxlen=5)

# === Main loop ===
frame_count = 0  # Counter to skip frames

recall = None
while True:
    frame = get_frame(source=source_type)
    if frame is None:
        print("Error: Could not retrieve frame.")
        break

    # frame_count += 1
    # if frame_count % 2 != 0:
    #     # Skip this frame
    #     continue

    # Resize to 32x32 and normalize
    resized_frame = cv2.resize(frame, (32, 32))
    normalized_frame = resized_frame.astype("float32") / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)

    # Predict
    pred_probs = model.predict(input_tensor)
    predicted_class = np.argmax(pred_probs, axis=1)[0]
    label = class_names[predicted_class]

    # Update prediction history
    prediction_window.append(label)

    # Determine if any class meets threshold
    confirmed_label = None
    if len(prediction_window) == 5:
        counts = Counter(prediction_window)
        for lbl, cnt in counts.items():
            if cnt >= 4:
                confirmed_label = lbl
                break

    # Display result
    display_text = f"Prediction: {confirmed_label if confirmed_label else 'Analyzing...'}"
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    if confirmed_label and recall != confirmed_label:
        recall = confirmed_label
        try:
            subprocess.Popen([
                ".venv/Scripts/python.exe", "speak_deeo_vision.py", confirmed_label
            ])
        except Exception as e:
            print("Failed to speak via NAO:", e)

    cv2.imshow("Classifier Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# === Cleanup ===
if cap:
    cap.release()
cv2.destroyAllWindows()
", "nao_identify_speak.py", confirmed_label
            ])
        except Exception as e:
            print("Failed to speak via NAO:", e)

    cv2.imshow("Classifier Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# === Cleanup ===
if cap:
    cap.release()
cv2.destroyAllWindows()
