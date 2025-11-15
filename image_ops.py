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


def crop_fraction(
    image: Image.Image,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Image.Image:
    width, height = image.size
    start_x, end_x = int(width * x_range[0]), int(width * x_range[1])
    start_y, end_y = int(height * y_range[0]), int(height * y_range[1])
    return image.crop((start_x, start_y, end_x, end_y))


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


def white_ratio(image: Image.Image, threshold: int = 200) -> float:
    gray = image.convert("L")
    arr = np.array(gray)
    white_pixels = np.sum(arr >= threshold)
    total_pixels = arr.size
    return white_pixels / float(total_pixels or 1)


@dataclass(frozen=True)
class RegionPreset:
    """窗口中常用区域配置。"""

    name: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


 

