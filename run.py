"""
程序主入口：包含截图和处理图片两个功能。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence
from types import SimpleNamespace

import yaml
import click

import cv2
import numpy as np
from paddleocr import PaddleOCR

# 读取配置文件
_config_path = Path(__file__).parent / "settings.yaml"
_config = yaml.safe_load(_config_path.read_text(encoding="utf-8")) or {}
_settings = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in _config.items()})
from src.producer import ProducerApp
from src.screen_capture import select_bbox


def _collect_text(ocr_result: Sequence) -> str:
    """从 OCR 结果中提取文本"""
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


def process_image_direct(image_path: Path, ocr: PaddleOCR) -> str:
    """直接处理图片（不使用 Celery）"""
    img = cv2.imread(str(image_path))
    if img is None:
        return ""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ocr_res = ocr.predict(img_rgb)
    return _collect_text(ocr_res)


def process_images_with_celery(image_dir: Path, celery_task) -> None:
    """使用 Celery 处理图片"""
    image_files = sorted(
        image_dir.glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    
    click.echo(f"找到 {len(image_files)} 张图片，开始处理...")
    
    for image_path in image_files:
        try:
            image_num = int(image_path.stem)
        except ValueError:
            click.echo(f"跳过无效文件名: {image_path}")
            continue
        
        # 直接传递图片路径，不再编码
        celery_task.delay(str(image_path), image_num, str(image_dir))
        click.echo(f"已提交任务: {image_path.name}")
    
    click.echo("所有任务已提交，请等待 Celery worker 处理...")


def process_images_direct(image_dir: Path, ocr: PaddleOCR) -> None:
    """直接处理图片（不使用 Celery）"""
    image_files = sorted(
        image_dir.glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    
    click.echo(f"找到 {len(image_files)} 张图片，开始处理...")
    
    for image_path in image_files:
        try:
            image_num = int(image_path.stem)
        except ValueError:
            click.echo(f"跳过无效文件名: {image_path}")
            continue
        
        click.echo(f"处理中: {image_path.name}...", nl=False)
        
        # 直接处理
        ocr_text = process_image_direct(image_path, ocr)
        
        # 保存文本
        text_file = image_dir / f"{image_num}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(ocr_text or "")
        
        click.echo(f" 完成 -> {text_file.name}")
    
    click.echo("所有图片处理完成！")


@click.group()
def cli() -> None:
    """程序主入口：截图和处理图片工具"""
    pass


@cli.command()
@click.option("--dir", "-d", default="results", help="截图保存目录")
def capture(dir: str) -> None:
    """开始截图，持续保存截图到指定目录"""
    try:
        click.echo("进入框选模式...")
        parsed_bbox = select_bbox()
        from src.executor import ActionExecutor
        from src.screen_capture import ScreenCapture
        screen_capture = ScreenCapture(parsed_bbox)
        action_executor = ActionExecutor(screen_capture)
        producer = ProducerApp(screen_capture, action_executor, save_dir=dir)
        click.echo("开始截图，按 Ctrl+C 停止...")
        producer.run_forever(interval_ms=_settings.capture.min_interval_ms)
    except KeyboardInterrupt:
        click.echo("\n截图已停止")
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--dir", "-d", default="results", help="图片目录路径")
@click.option("--celery", "-c", is_flag=True, help="使用 Celery 处理（需要启动 worker）")
def process(dir: str, celery: bool) -> None:
    """处理截图目录中的图片，进行 OCR 识别"""
    image_dir = Path(dir)
    
    if not image_dir.exists():
        click.echo(f"目录不存在: {image_dir}", err=True)
        sys.exit(1)
    
    if celery:
        # 使用 Celery 处理（需要先启动 worker）
        from src.worker import handle_capture_task
        click.echo("使用 Celery 模式处理，请确保已启动 worker")
        process_images_with_celery(image_dir, handle_capture_task)
    else:
        # 直接处理
        click.echo("初始化 OCR 引擎...")
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        process_images_direct(image_dir, ocr)


if __name__ == "__main__":
    cli()

