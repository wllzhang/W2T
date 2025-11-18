"""
Producer 端启动脚本，仅负责截图和保存截图。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

import time
import threading

import click
import cv2

# 读取配置文件
_config_path = Path(__file__).parent.parent / "settings.yaml"
_config = yaml.safe_load(_config_path.read_text(encoding="utf-8")) or {}
_settings = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in _config.items()})
from src.executor import ActionExecutor
from src.screen_capture import ScreenCapture, select_bbox

class ProducerApp:
    def __init__(
        self,
        screen_capture: ScreenCapture,
        action_executor: ActionExecutor,
        save_dir: str | Path | None = None,
    ) -> None:
        self.screen_capture = screen_capture
        self.action_executor = action_executor
        self.save_dir = Path(save_dir or "results")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_counter = 0
        self.counter_lock = threading.Lock()

    def run_forever(self, interval_ms: int = 500) -> None:
        interval = max(interval_ms, 100) / 1000.0

        while True:
            self.run_once()
            time.sleep(interval)

    def run_once(self) -> None:
        # 1. 截图
        self.screen_capture.capture()
        
        # 2. 保存截图（从1开始递增命名）
        with self.counter_lock:
            self.image_counter += 1
            image_num = self.image_counter
        
        image_path = self.save_dir / f"{image_num}.jpg"
        if self.screen_capture.last_array is not None:
            cv2.imwrite(str(image_path), cv2.cvtColor(self.screen_capture.last_array, cv2.COLOR_RGB2BGR))
            click.echo(f"截图已保存: {image_path}")
        
        # 3. 执行动作（如果需要）
        actions = [
            {"action": "move_to", "position": [0.8, 0.5]},
            {"action": "delay", "seconds": 1},
            {"action": "click"}
        ]
        self.action_executor.compose(actions)
        
        # 4. 显示预览
        # self._show_preview()
    
    def _show_preview(self) -> None:
        frame = self.screen_capture.last_array
        if frame is None:
            return
        cv2.imshow("Capture Preview", frame)
        cv2.waitKey(1)


@click.command()
def main() -> None:
    click.echo("进入框选模式...")
    parsed_bbox = select_bbox()
    screen_capture = ScreenCapture(parsed_bbox)
    action_executor = ActionExecutor(screen_capture)
    producer = ProducerApp(screen_capture, action_executor)
    producer.run_forever(interval_ms=_settings.capture.min_interval_ms)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        click.echo("producer stopped")


