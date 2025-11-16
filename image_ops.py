"""
图像通用操作，源自 `auto_answer/pre_handle/images_split.py` 的抽取版本。

提供：
- 任意输入到 PIL Image 的转换
- 指定比例裁剪
- 图像与 bytes/base64 的互转
- 白色像素比例等基础指标
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageSource = Union[str, Path, np.ndarray, Image.Image, ImageFile.ImageFile]


def load_image(source: ImageSource) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    if isinstance(source, np.ndarray):
        return Image.fromarray(source)
    if isinstance(source, (str, Path)):
        return Image.open(source)
    if isinstance(source, ImageFile.ImageFile):
        return source.copy()
    raise TypeError(f"Unsupported image source: {type(source)}")




def image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    with io.BytesIO() as buffer:
        image.save(buffer, format=fmt)
        return buffer.getvalue()

 


def encode_capture_payload(source: ImageSource, fmt: str = "PNG") -> bytes:
    image = load_image(source)
    return image_to_bytes(image, fmt=fmt)


def decode_capture_payload(payload: bytes) -> Optional[np.ndarray]:
    with io.BytesIO(payload) as buffer:
        try:
            with Image.open(buffer) as img:
                rgb = img.convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None
    return np.array(rgb)


 

 

