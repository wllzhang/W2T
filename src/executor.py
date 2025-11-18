"""
执行层：根据消费者返回的 ActionInstruction 执行鼠标键盘操作。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import pyautogui

from typing import Any, Dict, List, Optional

from src.screen_capture import ScreenCapture, select_bbox
import tkinter as tk
from PIL import Image, ImageTk


class ActionExecutor:
    def __init__(self, screen_capture: ScreenCapture) -> None:
        self.screen_capture = screen_capture
 
    # 最小动作：移动到指定坐标（支持相对[0,1]与绝对像素）
    def move_to(self, x: float, y: float) -> None:
        ax = float(x)
        ay = float(y)
        if 0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0:
            px, py = self.screen_capture.denormalize(ax, ay)
            pyautogui.moveTo(int(px), int(py))
        else:
            pyautogui.moveTo(int(ax), int(ay))
 
    def click(self) -> None:
        pyautogui.click()
 
    def scroll(self, offset: int) -> None:
        pyautogui.scroll(int(offset))
 
    def wait(self, seconds: float) -> None:
        time.sleep(float(seconds or 0.0))

    def drag_right(self, distance_px: int, duration: float = 0.2) -> None:
        """
        从当前鼠标位置开始，按住左键并向右拖动指定像素距离（屏幕像素）。
        :param distance_px: 向右拖动的像素数（>0 向右，<=0 不执行）
        :param duration: 拖动持续时间（秒），影响移动速度
        """
        if distance_px is None or distance_px <= 0:
            return
        duration = max(0.05, float(duration or 0.0))
        steps = max(1, min(abs(distance_px), 50))
        step_dx = distance_px / steps
        step_dt = duration / steps
        start_x, _ = pyautogui.position()
        pyautogui.mouseDown(button="left")
        try:
            for _ in range(steps):
                pyautogui.moveRel(step_dx, 0, duration=step_dt)
            current_x, _ = pyautogui.position()
            final_dx = start_x + distance_px - current_x
            if abs(final_dx) >= 1:
                pyautogui.moveRel(final_dx, 0, duration=0)
        finally:
            pyautogui.mouseUp(button="left")

    # 组合执行：每个元素仅表示一个动作：
    #     [
    # {"action": "move_to", "position": [0.2, 0.3]},
    # {"action": "delay", "seconds": 0.05},
    # {"action": "click"},
    # {"action": "scroll", "offset": -240},
    # {"action": "wait", "seconds": 0.2},
    # {"action": "move_to", "x": 800, "y": 500},
    # {"action": "click"},
    # {"action": "drag_right", "distance": 200, "duration": 0.25}
    # ]
    def compose(self, actions: Optional[List[Dict[str, Any]]]) -> None:
        if not actions:
            return
        for item in actions:
            name = (item.get("action") or "").lower()

            if name == "move_to":
                if "position" in item and item["position"] and len(item["position"]) >= 2:
                    x, y = float(item["position"][0]), float(item["position"][1])
                elif "x" in item and "y" in item:
                    x, y = float(item["x"]), float(item["y"])
                else:
                    raise ValueError("move_to requires position=[x,y] or x,y")
                self.move_to(x, y)
            elif name == "click":
                # 始终在当前位置点击，不根据传入坐标移动
                self.click()
            elif name == "scroll":
                offset = int(item.get("offset", 0) or 0)
                self.scroll(offset)
            elif name == "wait":
                seconds = float(item.get("seconds", 0.0) or 0.0)
                self.wait(seconds)
            elif name == "delay":
                seconds = float(item.get("seconds", 0.0) or 0.0)
                self.wait(seconds)
            elif name == "drag_right":
                distance_px = int(item.get("distance", 0) or 0)
                duration = float(item.get("duration", 0.2) or 0.2)
                if distance_px > 0:
                    self.drag_right(distance_px=distance_px, duration=duration)
            else:
                raise ValueError(f"Unsupported action name: {name}")

    def run_demo(self) -> None:
        """
        简单效果演示：
        1) 移动到 ROI 中心并点击
        2) 向上滚动一点并等待
        3) 移动到 ROI 左下角附近，然后向右拖动 200 像素
        """
        demo_actions: List[Dict[str, Any]] = [
            {"action": "move_to", "position": [0.5, 0.5]},
            {"action": "delay", "seconds": 0.3},
            {"action": "click"},
            {"action": "delay", "seconds": 0.2},
            {"action": "scroll", "offset": -240},
            {"action": "wait", "seconds": 0.2},
            {"action": "move_to", "position": [0.2, 0.8]},
            {"action": "delay", "seconds": 0.2},
            {"action": "drag_right", "distance": 200, "duration": 0.25},
        ]
        self.compose(demo_actions)

     

 

if __name__ == "__main__":
    executor = ActionExecutor(ScreenCapture(select_bbox()))
    executor.run_demo()

