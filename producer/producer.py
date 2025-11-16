"""
Producer 端启动脚本，仅保留三个核心类：
1. ScreenCapture：负责截图与任务封装
2. ActionExecutor：根据识别结果执行动作
3. ProducerApp：调度截图 -> Celery 任务 -> 执行动作
"""

from __future__ import annotations

import time
import click
import cv2

from settings import settings
from consumer.worker import handle_capture_task
from executors.executor import ActionExecutor
from producer.screen_capture import ScreenCapture, select_bbox

class ProducerApp:
    def __init__(
        self,
        screen_capture: ScreenCapture,
        action_executor: ActionExecutor,
        celery_task,
    ) -> None:
        self.screen_capture = screen_capture
        self.action_executor = action_executor
        self.celery_task = celery_task

    def run_forever(self, interval_ms: int = 500) -> None:
        interval = max(interval_ms, 100) / 1000.0

        while True:
            self.run_once()
            time.sleep(interval)

    def run_once(self) -> None:
        frame_payload = self.screen_capture.capture()
        # async_result = self.celery_task.delay(frame_payload)
        # result = async_result.get(timeout=settings.queue.task_timeout_seconds)
        actions =  [
            {"action": "move_to", "position": [0.8, 0.5]},
            {"action": "delay", "seconds": 1},
            {"action": "click"}
        ]
        self.action_executor.compose( actions )
        self._show_preview()

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
    producer = ProducerApp(screen_capture, action_executor, handle_capture_task)
    producer.run_forever(interval_ms=settings.capture.min_interval_ms)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        click.echo("producer stopped")

