"""
执行层：根据消费者返回的 ActionInstruction 执行鼠标键盘操作。
"""

from __future__ import annotations

import time

import pyautogui

from typing import Any, Dict, List, Optional

from producer.screen_capture import ScreenCapture


class ActionExecutor:
    def __init__(self, screen_capture: ScreenCapture) -> None:
        self.screen_capture = screen_capture

    def run(self, actions: Optional[List[Dict[str, Any]]]) -> None:
        if not actions:
            return
        for instruction in actions:
            self._execute_instruction(instruction)

    def _execute_instruction(self, instruction: Dict[str, Any]) -> None:
        action = instruction.get("action", "noop")
        delay = float(instruction.get("delay", 0.0) or 0.0)
        if delay:
            time.sleep(delay)
        if action == "click":
            self._handle_click(instruction)
        elif action == "scroll":
            self._handle_scroll(instruction)
        elif action == "wait":
            time.sleep(delay or 0.1)
        elif action == "noop":
            return
        else:
            raise ValueError(f"Unsupported action: {action}")

    def _handle_click(self, instruction: Dict[str, Any]) -> None:
        position = instruction.get("position")
        if not position:
            raise ValueError("Click action requires position")
        x, y = self.screen_capture.denormalize(*position)
        pyautogui.moveTo(x, y)
        pyautogui.click()

    def _handle_scroll(self, instruction: Dict[str, Any]) -> None:
        position = instruction.get("position") or [0.0, -1.0]
        offset = position[1] if len(position) > 1 else -1.0
        pyautogui.scroll(int(offset * 120))

