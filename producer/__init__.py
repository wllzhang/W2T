"""
Producer 侧核心组件：

- `ScreenCapture`：负责截屏与 TaskPayload 构造
- `ActionExecutor`：执行消费者返回的动作
"""

from .screen_capture import ScreenCapture


__all__ = ["ScreenCapture"]