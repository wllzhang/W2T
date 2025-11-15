from __future__ import annotations
import logging
"""
消费者 Celery 占位实现。

真实逻辑尚未完成时，用单个 Celery 任务模拟识别 + 动作决策。
"""

"""
消费者端：本地 OCR Worker。

参考 auto_answer 中的 PaddleOCR 用法，实现基础识别服务。
"""

import eventlet

eventlet.monkey_patch()

import sys
from typing import List

import numpy as np
from celery import Celery
from typing import List, Sequence
from image_ops import decode_capture_payload
from settings import settings
app = Celery(
    "w2t_consumer",
    broker=settings.queue.broker_url,
    backend=settings.queue.result_backend,
)
app.conf.update(
    accept_content=["pickle", "json"],
    task_serializer="pickle",
    result_serializer="pickle",
)

logger = logging.getLogger(__name__)

from paddleocr import PaddleOCR

_ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False)
def _collect_text(ocr_result: Sequence) -> str:
    if not ocr_result:
        return ""
    entry = ocr_result[0]
    if isinstance(entry, dict):
        texts = entry.get("rec_texts") or []
        return "".join(str(t) for t in texts)

    text_segments: List[str] = []
    for block in entry:
        if not block or not isinstance(block, (list, tuple)) or len(block) < 2:
            continue
        info = block[1]
        if not info or not isinstance(info, (list, tuple)):
            continue
        text_segments.append(str(info[0]))
    return "".join(text_segments)

def _ocr_image(image_payload: bytes) -> str:
    frame = decode_capture_payload(image_payload)
    if frame is None or not isinstance(frame, np.ndarray):
        return ""
    ocr_res = _ocr.predict(frame)
    
    return _collect_text(ocr_res)


@app.task(name=settings.queue.default_routing_key)
def handle_capture_task(image_payload: bytes) -> str:
    """本地 OCR 任务，返回识别文本。"""

    return _ocr_image(image_payload)


def _default_worker_args() -> List[str]:
    return [
        "worker",
        "-l",
        "debug",
        "-P",
        "eventlet",
        "-c",
        "1",
    ]


if __name__ == "__main__":
    argv = sys.argv[1:] or _default_worker_args()
    app.worker_main(argv)