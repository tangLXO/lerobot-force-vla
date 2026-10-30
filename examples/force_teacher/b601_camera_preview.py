"""Explicit camera-only preview; resources remain retained until closure succeeds."""

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.force_control.demonstration import TeacherRecordConfig


class PreviewSession:
    def __init__(self, factory=OpenCVCamera):
        self.factory = factory
        self.cameras = {}

    def start(self, raw):
        self.close()
        configs = TeacherRecordConfig(root="preview", task="preview", camera_configs=raw).camera_objects()
        try:
            for role, config in configs.items():
                camera = self.factory(config)
                self.cameras[role] = camera
                camera.connect()
        except BaseException:
            self.close()
            raise

    def close(self):
        errors = []
        for role, camera in tuple(self.cameras.items()):
            try:
                if camera.has_resources:
                    camera.disconnect()
            except Exception as exc:
                errors.append(exc)
            else:
                del self.cameras[role]
        if errors:
            raise ExceptionGroup("预览资源尚未关闭，不能交接正式采集", errors)
