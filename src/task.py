import threading
from typing import Callable

from loguru import logger


class SingleThreadTask:

    def __init__(self):
        self._task = None
        self._stop = False

    def is_alive(self) -> bool:
        return self._task is not None and self._task.is_alive()

    def _job(self, target, *args):
        try:
            while not self._stop:
                target(*args)
        except Exception as e:
            logger.exception(
                "Unhandled exception occurred in SingleThreadTask."
            )

    def start(self, target: Callable, args=None):
        if self.is_alive():
            logger.warning('Thread has already been started.')
            return
        if args is None:
            args = []
        self._task = threading.Thread(target=self._job, args=[target, *args])
        self._task.start()

    def stop(self):
        if not self.is_alive():
            logger.warning('Thread has already been stopped.')
            return
        self._stop = True
        self._task.join()
        self._stop = False  # back to the initial state
