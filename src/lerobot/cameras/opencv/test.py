from .camera_opencv import OpenCVCamera
from .configuration_opencv import OpenCVCameraConfig
import cv2

cameras_indexes = [0,2]

def open_cameras():
    cameras = []
    for index in cameras_indexes:
        config = OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=30, fourcc_name='MJPG')
        camera = OpenCVCamera(config)
        camera.connect()
        cameras.append(camera)
    return cameras
 
def test_cameras(cameras):
    for camera in cameras:
        for i in range(10):
            frame = camera.read()
            print(frame.shape)

def close_cameras(cameras):
    for camera in cameras:
        camera.disconnect()

if __name__ == "__main__":
    cameras = open_cameras()
    test_cameras(cameras)
    close_cameras(cameras)

