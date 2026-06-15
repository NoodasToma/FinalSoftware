import cv2
import numpy as np
import os
import yaml
from typing import Tuple, Optional
from .camera_driver_abs import CameraDriverAbs


class CameraDriver(CameraDriverAbs):
    def _build_gstreamer_pipeline(self) -> str:
        # nvarguscamerasrc only outputs native sensor modes (1280x720, 1640x1232, etc.)
        # Requesting a non-native size (e.g. 640x480) in the source caps causes caps
        # negotiation failure. Let the source pick a native mode by framerate, then
        # scale to the desired output size via nvvidconv.
        pipeline = (
          f"nvarguscamerasrc  ! "
          f"video/x-raw(memory:NVMM), format=NV12, framerate={self.framerate}/1 ! "
          f"nvvidconv ! "
          f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
          f"videoconvert ! "
          f"appsink drop=true sync=false"
        )


        return pipeline



    def _initialize_camera(self):
        import time

        pipeline = self._build_gstreamer_pipeline()
        print(f"[JetsonCamera] Pipeline: {pipeline}")

        # OUTER retry: when a previous task is slow to release the CSI camera (a
        # restart race) nvargus refuses a new CaptureSession and the FIRST built
        # pipeline yields no frames forever. Retrying reads on that dead pipeline
        # never recovers — we must tear the whole VideoCapture down and REBUILD a
        # fresh one, giving the old session time to drain. A truly wedged
        # nvargus-daemon still needs a daemon restart / reboot, but this rides out
        # the common "redeployed too fast" case without manual intervention.
        outer_attempts = 6
        for attempt in range(1, outer_attempts + 1):
            self._device = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self._device.isOpened():
                # isOpened() can be True even when GStreamer fails internally, so
                # confirm real frames flow (nvargus needs a moment to warm up).
                for _ in range(10):
                    ret, _ = self._device.read()
                    if ret:
                        int(self._device.get(cv2.CAP_PROP_FRAME_WIDTH))
                        int(self._device.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if attempt > 1:
                            print(f"[JetsonCamera] Acquired camera on attempt {attempt}")
                        return
                    time.sleep(0.3)

            # This attempt failed — release and rebuild after a growing pause so
            # the previous holder's session can fully release.
            if self._device is not None:
                self._device.release()
                self._device = None
            if attempt < outer_attempts:
                wait = 1.0 + attempt          # 2s, 3s, 4s, 5s, 6s
                print(f"[JetsonCamera] No frames (attempt {attempt}/{outer_attempts}); "
                      f"rebuilding pipeline in {wait:.0f}s "
                      f"(prior task may still hold the camera)...")
                time.sleep(wait)

        raise RuntimeError(
            "Camera opened but returned no frames after warm-up — the CSI camera "
            "is held by another process or nvargus-daemon is wedged. Fix on the "
            "bot: `sudo systemctl restart nvargus-daemon` (or reboot), then start once.")

    def _capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._device is None:
            return False, None

        ret, frame = self._device.read()
        return (True, frame) if ret else (False, None)

    def _release_camera(self):
        if self._device is not None:
            self._device.release()
            self._device = None
