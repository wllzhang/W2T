"""
全局配置：从根目录的 settings.yaml 读取。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

CONFIG_PATH = Path(__file__).with_name("settings.yaml")


@dataclass
class QueueSettings:
    broker_url: str
    result_backend: str
    default_routing_key: str
    task_timeout_seconds: float


@dataclass
class CaptureSettings:
    min_interval_ms: int
    enable_foreground_check: bool


@dataclass
class OCRSettings:
    channels: List[str]
    timeout_seconds: float


@dataclass
class Settings:
    queue: QueueSettings
    capture: CaptureSettings
    ocr: OCRSettings


def _merge(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    data = defaults.copy()
    data.update(overrides or {})
    return data


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    if not path.exists():
        raise FileNotFoundError(f"配置文件 {path} 不存在")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queue = QueueSettings(**raw["queue"])
    capture = CaptureSettings(**raw["capture"])
    ocr = OCRSettings(**raw["ocr"])
    return Settings(queue=queue, capture=capture, ocr=ocr)


settings = load_settings()

# print(settings)