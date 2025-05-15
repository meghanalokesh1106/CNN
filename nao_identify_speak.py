# camera_speak.py (Python 2.7 compatible for NAO)
from naoqi import ALProxy
import time
import base64
import requests
import json
import numpy as np
import cv2

# ---- NAO Configuration ----
NAO_IP = "172.18.16.45"
PORT = 9559
SERVER_URL = "http://<YOUR_PC_IP>:5000/predict"  # Flask server URL

# ---- Proxies ----
video = ALProxy("ALVideoDevice", NAO_IP, PORT)
tts = ALProxy("ALTextToSpeech", NAO_IP, PORT)

# ---- Subscribe to camera ----
resolution = 2    # VGA
colorSpace = 11   # RGB
fps = 5

camera_name = "nao_object_detect"
video.unsubscribeAll()
capture_id = video.subscribeCamera(camera_name, 0, resolution, colorSpace, fps)

# ---- Capture and Predict ----
print("Capturing image...")
nao_image = video.getImageRemote(capture_id)
video.unsubscribe(camera_name)

if nao_image:
    width = nao_image[0]
    height = nao_image[1]
    array = nao_image[6]

    image_np = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))

    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', image_np)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Send to server
    try:
        response = requests.post(SERVER_URL, json={"image": jpg_as_text}, timeout=5)
        if response.status_code == 200:
            prediction = response.json().get("prediction", "something")
            print("Prediction:", prediction)
            tts.say("I think this is a " + prediction)
        else:
            print("Server error:", response.status_code)
            tts.say("Sorry, I could not identify the object.")
    except Exception as e:
        print("Error sending request:", e)
        tts.say("I cannot connect to the server.")
else:
    tts.say("Camera error. Could not get the image.")
