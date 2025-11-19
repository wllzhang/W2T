# W2T - Window to Text

截图和 OCR 识别工具，支持实时截图、批量处理和文本合并。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 Redis（编辑 settings.yaml）
# 启动 Celery worker（可选，用于并行处理）
python src/worker.py

# 截图
python run.py capture -d results

# OCR 处理（直接模式）
python run.py process -d results

# OCR 处理（Celery 模式，需先启动 worker）
python run.py process -d results -c

# 合并文本文件
python run.py mergetxt -d results
```

## 命令说明

### 截图
```bash
python run.py capture -d <目录>
```
- 框选区域：鼠标拖拽选择，右键确认，ESC 取消
- 自动保存：`1.jpg`, `2.jpg`, `3.jpg`...
- 停止：按 `Ctrl+C`

### OCR 处理
```bash
# 直接模式（顺序处理）
python run.py process -d <目录>

# Celery 模式（并行处理，需先启动 worker）
python run.py process -d <目录> -c
```

### 合并文本
```bash
python run.py mergetxt -d <目录>
```
- 按文件名序号排序合并所有 `.txt` 文件
- 输出：`merged.txt`

### 图片分类（深度学习）
```bash
# 训练分类模型
python -m src.deep_classifier train

# 预测单张图片
python -m src.deep_classifier predict <图片路径>

# 预测目录中的所有图片
python -m src.deep_classifier predict <目录>
```
- 使用 CNN 模型将图片分为两类
- 需要先训练模型（使用 `results/train/` 目录中的图片）
- 模型保存位置：`results/classifier_model.pth`

## 配置

编辑 `settings.yaml`：

```yaml
queue:
  broker_url: "redis://:密码@主机:6379/0"
  result_backend: "redis://:密码@主机:6379/1"

capture:
  min_interval_ms: 100  # 截图间隔（毫秒）
```

**注意**：Redis 密码中的特殊字符需 URL 编码（`@` → `%40`，`#` → `%23`）

## Celery Worker

```bash
# 启动 worker（使用 prefork 多进程模式，充分利用多核）
python src/worker.py

# 或手动指定参数
celery -A src.worker worker -P prefork -c 8 -l info
```

**推荐配置**：
- 16 核 CPU：`-c 8` 或 `-c 12`
- 8 核 CPU：`-c 4` 或 `-c 6`

## 文件结构

```
W2T/
├── run.py              # 主入口
├── settings.yaml       # 配置文件
├── src/
│   ├── worker.py       # Celery worker
│   ├── producer.py     # 截图生产者
│   ├── screen_capture.py
│   ├── executor.py
│   └── deep_classifier.py  # 图片分类模型
└── results/            # 输出目录
```

## 工作流程

1. **截图** → `1.jpg`, `2.jpg`, `3.jpg`...
2. **OCR 处理** → `1.txt`, `2.txt`, `3.txt`...
3. **合并文本** → `merged.txt`
4. **图片分类**（可选）→ 使用深度学习模型对图片进行分类
