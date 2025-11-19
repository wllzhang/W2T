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

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

import eventlet

eventlet.monkey_patch()

from typing import List, Sequence

import cv2
import numpy as np
from celery import Celery

# 读取配置文件
_config_path = Path(__file__).parent.parent / "settings.yaml"
_config = yaml.safe_load(_config_path.read_text(encoding="utf-8")) or {}
_settings = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in _config.items()})

app = Celery(
    "w2t_consumer",
    broker=_settings.queue.broker_url,
    backend=_settings.queue.result_backend,
)

logger = logging.getLogger(__name__)

from paddleocr import PaddleOCR

_ocr_instance = None

def _get_ocr() -> PaddleOCR:
    """懒加载 OCR 实例，只在首次调用时初始化"""
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("初始化 PaddleOCR 引擎...")
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False
        )
    return _ocr_instance

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

def _ocr_image(image_path: str | Path) -> str:
    """直接读取图片文件进行 OCR"""
    img = cv2.imread(str(image_path))
    if img is None:
        return ""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ocr = _get_ocr()
    ocr_res = ocr.predict(img_rgb)
    return _collect_text(ocr_res)


@app.task(name=_settings.queue.default_routing_key)
def handle_capture_task(image_path: str, image_num: int, save_dir: str) -> str:
    """本地 OCR 任务，处理完后保存文本文件，文件名与图片序号对应。"""
    ocr_text = _ocr_image(image_path)
    
    # 按照图片序号保存文本文件
    save_path = Path(save_dir)
    text_file = save_path / f"{image_num}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(ocr_text or "")
    
    logger.info(f"OCR 完成：图片 {image_num}.jpg -> {image_num}.txt")
    return ocr_text


def _default_worker_args() -> List[str]:
    return [
        "worker",
        "-l",
        "info",
        "-P",
        "eventlet",
        "-c",
        "13",
    ]


if __name__ == "__main__":
    argv = sys.argv[1:] or _default_worker_args()
    app.worker_main(argv)

