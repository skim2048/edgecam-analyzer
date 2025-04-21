import os
import gc
import json
from concurrent import futures
from contextlib import asynccontextmanager

import cv2
import grpc
import torch
import numpy as np
import ultralytics
from loguru import logger
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from payload import frame_pb2, frame_pb2_grpc


# ---------- TODO TEMPORARY ----------
if not "model.pt" in os.listdir('.'):
    from src.download import download_facedet
    download_facedet()
YOLO = ultralytics.YOLO("model.pt")
# ------------- TODO END -------------


class AnalyzerService(frame_pb2_grpc.AnalyzerServiceServicer):

    def Deidentify(self, request, context):
        frame = np.frombuffer(request.frame, dtype=np.uint8).reshape(request.shape)
        frame = frame.copy()
        # cv2.rectangle(frame, (100, 100), (200, 200), (0, 0, 255), 2)
        # logger.info(f'frame.shape -> {frame.shape}')
        # ---------- TODO TEMPORARY ----------
        boxes = YOLO.predict(frame, verbose=False)
        boxes = YOLO.track(frame, persist=True, verbose=False)[0].boxes.data.cpu().numpy()
        xyxy = boxes[:, :4].astype(int)
        for i in range(boxes.shape[0]):
            p1 = tuple(xyxy[i][:2])
            p2 = tuple(xyxy[i][2:])
            x1, y1 = p1
            x2, y2 = p2
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            frame[y1:y2, x1:x2] = blurred_roi
        frame = np.array(frame)
        # ------------- TODO END -------------
        return frame_pb2.Frame(shape=list(frame.shape), frame=frame.tobytes())


VIDEO_ANALYZER = None

with open("settings.json", 'r') as f:
    SETTINGS = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global VIDEO_ANALYZER
    global YOLO
    VIDEO_ANALYZER = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    frame_pb2_grpc.add_AnalyzerServiceServicer_to_server(AnalyzerService(), VIDEO_ANALYZER)
    VIDEO_ANALYZER.add_insecure_port("0.0.0.0:12932")
    VIDEO_ANALYZER.start()
    logger.info("gRPC Server Started.")
    yield
    VIDEO_ANALYZER.stop(grace=0)
    if next(YOLO.parameters()).device.type == 'cuda':
        YOLO.to('cpu')
        torch.cuda.empty_cache()
    del YOLO
    gc.collect()
    logger.info("gRPC Server Stopped.")


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS["allowed_cors"],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],)


# @app.post('/hello')
# async def say_hello():
#     try:
#         return {"status": "hello"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app=SETTINGS["app"],
        host=SETTINGS["host"],
        port=SETTINGS["port"]
    )
