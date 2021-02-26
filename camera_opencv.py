import os
import cv2

from base_camera import BaseCamera

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


class Camera(BaseCamera):
    video_source = 0
    current_face_num = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # 获取摄像头拍摄到的画面
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            Camera.current_face_num = len(faces)
            cv2.putText(img, 'Number of faces detected:' + str(Camera.current_face_num), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            for (x, y, w, h) in faces:
                # 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
                face_area = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(face_area)
                if len(eyes):
                    # 画出人脸，绿色
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
                    for (ex, ey, ew, eh) in eyes:
                        # 画出人眼框，蓝色，画笔宽度为1
                        cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
                else:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
