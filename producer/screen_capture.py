"""
屏幕截图模块：负责 ROI 配置与 TaskPayload 生成。
"""

from __future__ import annotations

import time
import uuid
from typing import Optional, Tuple

import numpy as np
import pyautogui
import tkinter as tk
from PIL import ImageTk

from image_ops import encode_capture_payload
from settings import settings


def select_bbox() -> Tuple[int, int, int, int]:
    screenshot = pyautogui.screenshot()
    root = tk.Tk()
    root.title("请选择监控区域，右键确认/ESC取消")
    root.attributes("-fullscreen", True)

    canvas = tk.Canvas(root, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    photo = ImageTk.PhotoImage(screenshot)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    selection = {"start": (0, 0), "end": (0, 0), "done": False}
    rect = canvas.create_rectangle(0, 0, 0, 0, outline="red", width=2)

    def on_press(event):
        selection["start"] = (event.x, event.y)
        selection["end"] = (event.x, event.y)
        canvas.coords(rect, event.x, event.y, event.x, event.y)

    def on_move(event):
        selection["end"] = (event.x, event.y)
        canvas.coords(rect, selection["start"][0], selection["start"][1], event.x, event.y)

    def on_release(event):
        selection["end"] = (event.x, event.y)
        canvas.coords(rect, selection["start"][0], selection["start"][1], event.x, event.y)

    def on_right_click(event):
        selection["done"] = True
        root.destroy()

    def on_escape(event):
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<ButtonRelease-3>", on_right_click)
    root.bind("<Escape>", on_escape)

    root.mainloop()

    if not selection["done"]:
        raise RuntimeError("未选择窗口区域，已取消")

    start_x, start_y = selection["start"]
    end_x, end_y = selection["end"]
    left = min(start_x, end_x)
    top = min(start_y, end_y)
    right = max(start_x, end_x)
    bottom = max(start_y, end_y)
    return (left, top, right, bottom)


class ScreenCapture:
    def __init__(self, bbox: Tuple[int, int, int, int]) -> None:
        self.bbox = bbox  # (left, top, right, bottom)
        self.last_array: Optional[np.ndarray] = None

    def capture(self):
        left, top, right, bottom = self.bbox
        width = right - left
        height = bottom - top
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        self.last_array = np.array(screenshot)
        return encode_capture_payload(screenshot, fmt="PNG")
       

    def normalize(self, x: int, y: int) -> Tuple[float, float]:
        left, top, right, bottom = self.bbox
        width = right - left
        height = bottom - top
        return (x - left) / width, (y - top) / height

    def denormalize(self, nx: float, ny: float) -> Tuple[int, int]:
        left, top, right, bottom = self.bbox
        width = right - left
        height = bottom - top
        return int(left + nx * width), int(top + ny * height)

