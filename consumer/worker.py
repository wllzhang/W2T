from __future__ import annotations
"""
消费者 Celery 占位实现。

真实逻辑尚未完成时，用单个 Celery 任务模拟识别 + 动作决策。
"""

"""
消费者端：本地 OCR Worker。

参考 auto_answer 中的 PaddleOCR 用法，实现基础识别服务。
"""



import base64
from typing import List

from celery import Celery


from settings import settings
app = Celery(
    "w2t_consumer",
    broker=settings.queue.broker_url,
    backend=settings.queue.result_backend,
)

# from paddleocr import PaddleOCR

# _ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang="ch",
#     use_gpu=False,
#     show_log=False,
# )


# def _ocr_image(image_b64: str) -> str:
#     image_bytes = base64.b64decode(image_b64)
#     ocr_res = _ocr.ocr(image_bytes)
#     if not ocr_res or not ocr_res[0]:
#         return ""
#     text_segments: List[str] = []
#     for line in ocr_res[0]:
#         if not line or not line[1]:
#             continue
#         text_segments.append(line[1][0])
#     return "".join(text_segments)


@app.task(name=settings.queue.default_routing_key)
def handle_capture_task(image_b64: str) -> str:
    """本地 OCR 任务，返回识别文本。"""

    return "hello world"