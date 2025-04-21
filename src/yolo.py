import gc
from typing import Callable

import numpy as np
import torch
import ultralytics

from src.buffer import SingleSlotBuffer
from src.task import SingleThreadTask


class YoloDetector:

    def __init__(self, pt='yolov8n.pt', tracking=False):
        self._model = ultralytics.YOLO(pt)
        self._tracking = tracking

    def predict(self, img: np.ndarray) -> np.ndarray:
        if self._tracking:
            results = self._model.track(img, persist=True, verbose=False)
        else:
            results = self._model.predict(img, verbose=False)
        return { 'boxes' : results[0].boxes.data.cpu().numpy() }

    def release(self):
        if next(self._model.parameters()).device.type == 'cuda':
            self._model.to('cpu')
            torch.cuda.empty_cache()
        del self._model
        gc.collect()


class Yolo:

    def __init__(self, pt='yolov8n.pt', tracking=False):
        self._detector = YoloDetector(pt, tracking)
        self._buffer = SingleSlotBuffer()
        self._buffer_task = SingleThreadTask()
        self._read_frame = None  # Callable

    def _buffering(self):
        frame = self._read_frame()
        results = self._detector.predict(frame)
        self._buffer.put([frame, results])

    def predict(self, get_image: Callable):
        self._read_frame = get_image
        self._buffer_task.start(self._buffering)

    def release(self):
        self._buffer_task.stop()
        self._detector.release()

    def read_preds(self, timeout_sec=30.0) -> list:
        return self._buffer.get(timeout_sec)
