"""
程序主入口：包含截图和处理图片两个功能。
"""

from __future__ import annotations

import sys
import time
import atexit
import asyncio
import threading
from pathlib import Path
from typing import List, Sequence
from types import SimpleNamespace

import yaml
import click
import requests
import aiohttp
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
    
    # 添加清理逻辑，避免退出时的 multiprocessing ResourceTracker 异常
    def cleanup():
        """清理 Celery 连接，避免退出时的异常"""
        try:
            # 通过任务对象访问 Celery 应用
            celery_app = celery_task.app
            # 关闭连接池和 broker 连接
            if hasattr(celery_app, 'pool') and celery_app.pool:
                try:
                    celery_app.pool.close()
                except Exception:
                    pass
            # 关闭 broker 连接
            if hasattr(celery_app, 'broker_connection') and celery_app.broker_connection:
                try:
                    celery_app.broker_connection.close()
                except Exception:
                    pass
        except Exception:
            pass
    
    # 注册退出时的清理函数
    atexit.register(cleanup)
    # 给连接一点时间完成，避免立即退出导致的资源清理问题
    time.sleep(0.1)


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
@click.option("--dir", "-d", required=True, help="截图保存目录")
@click.option("--times", "-t", default=0, help="截图次数")
def capture(dir: str,times: int) -> None:
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
        producer.run_forever(interval_ms=_settings.capture.min_interval_ms,times=times)
    except KeyboardInterrupt:
        click.echo("\n截图已停止")
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--dir", "-d",required=True, help="图片目录路径")
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


@cli.command()
@click.option("--dir", "-d" ,required=True, help="txt文件目录")
def mergetxt(dir: str) -> None:
    """合并目录中的所有 txt 文件内容到一个 merged.txt 文件"""
    txt_dir = Path(dir)
    if not txt_dir.exists():
        click.echo(f"目录不存在: {txt_dir}", err=True)
        sys.exit(1)
    
    # 1.获取当前目录下的所有txt文件（排除 merged.txt）
    # 2.按文件名排序
    # 3.读取文件内容
    # 4.合并文件内容
    # 5.保存合并后的内容到新的txt文件
    txt_files = sorted(
        Path(dir).glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit()  else 0
    )
    save_file = Path(dir) / "merged.txt"
    with open(save_file, "w", encoding="utf-8") as m_f:
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as t_f:
                m_f.write(t_f.read())
                m_f.write("\n")
    click.echo(f"合并完成！结果已保存到: {save_file}")
    # 自动翻译合并后的文件
    translate_file(str(save_file), 10)

async def translate_lines_async(lines: List[str], max_concurrent: int = 10, 
                                progress_callback=None) -> List[str]:
    """使用协程并发翻译多行文本"""
    semaphore = asyncio.Semaphore(max_concurrent)  # 限制并发数，避免过多请求
    
    async def translate_line(session: aiohttp.ClientSession, line: str) -> str:
        """异步翻译单行文本"""
        async with semaphore:  # 限制并发数
            try:
                async with session.post(
                    "https://api.deeplx.org/xxk7RND_15cVQlxL0aMo-3cNUEVNX0GoMY-rZcy3nDY/translate",
                    json={"text": line, "target_lang": "EN", "source_lang": "ZH"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    if progress_callback:
                        progress_callback()
                    return result.get('data', line)  # 如果失败，返回原文
            except Exception as e:
                click.echo(f"\n翻译错误: {e}", err=True)
                if progress_callback:
                    progress_callback()
                return line  # 出错时返回原文
    
    async with aiohttp.ClientSession() as session:
        tasks = [translate_line(session, line) for line in lines]
        results = await asyncio.gather(*tasks)
        return results


def translate_file(file: str, concurrent: int = 10) -> None:
    """翻译文件的核心逻辑（可被其他函数调用）"""
    file_path = Path(file)
    if not file_path.exists():
        click.echo(f"文件不存在: {file}", err=True)
        return
    
    save_file = file.replace(".txt", "_en.txt")
    
    # 读取所有行
    click.echo("读取文件...")
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 预处理：清理空行和格式化
    processed_lines = [line for line in lines if line.strip()]
    if not processed_lines:
        click.echo("文件中没有可翻译的内容", err=True)
        return
    
    click.echo(f"找到 {len(processed_lines)} 行需要翻译，开始翻译（并发数: {concurrent}）...")
    
    # 使用 click.progressbar 显示进度
    with click.progressbar(length=len(processed_lines), label="翻译中...", show_percent=True) as bar:
        progress_lock = threading.Lock()
        
        def update_progress():
            """线程安全的进度更新"""
            with progress_lock:
                bar.update(1)
        
        # 创建异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(translate_lines_async(processed_lines, concurrent, update_progress))
        finally:
            loop.close()
    
    # 保存结果
    with open(save_file, "w", encoding="utf-8") as f:
        for result in results:
            if result.strip() == "":
                continue
            f.write(result)
    
    click.echo(f"\n翻译完成！结果已保存到: {save_file}")


@cli.command()
@click.option("--file", "-f", required=True, help="txt文件路径")
@click.option("--concurrent", "-c", default=10, help="并发数（默认10）")
def translate(file: str, concurrent: int) -> None:
    """使用协程并发翻译文本文件"""
    translate_file(file, concurrent)

if __name__ == "__main__":
    cli()
