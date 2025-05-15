# Python 2.7 compatible

import cv2
import numpy as np
import base64
from naoqi import ALProxy
import sys


nao_ip = "172.18.16.45"

def get_nao_camera_frame(cameraId=0):
    camProxy = ALProxy("ALVideoDevice", nao_ip, 9559)
    resolution = 2  # VGA
    colorSpace = 13  # RGB
    fps = 30
    clientName = "camera_client"

    videoClient = camProxy.subscribeCamera(clientName, cameraId, resolution, colorSpace, fps)
    try:
        image = camProxy.getImageRemote(videoClient)
        if image is None:
            return None
        npimg = np.fromstring(image[6], dtype=np.uint8)
        frame = npimg.reshape((image[1], image[0], 3))
        return frame
    finally:
        camProxy.unsubscribe(videoClient)



if __name__ == "__main__":
    frame = get_nao_camera_frame()
    if frame is not None:
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            encoded = base64.b64encode(buffer)
            sys.stdout.write(encoded)
            sys.stdout.flush()
