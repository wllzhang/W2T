# W2T - Window to Text

截图和 OCR 识别工具，支持实时截图和批量图片处理。

## 功能特性

- 📸 **实时截图**：支持框选区域并持续截图
- 🔍 **OCR 识别**：使用 PaddleOCR 进行文字识别
- ⚡ **并行处理**：支持 Celery 分布式处理
- 📝 **自动保存**：截图和识别结果自动保存

## 安装

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖

- `click` - 命令行界面
- `pyautogui` - 屏幕截图和鼠标控制
- `opencv-python` - 图像处理
- `paddleocr` - OCR 引擎
- `celery` - 分布式任务队列
- `redis` - 消息队列后端
- `pyyaml` - 配置文件解析

## 配置

编辑 `settings.yaml` 配置文件：

```yaml
queue:
  broker_url: "redis://:密码@主机:端口/数据库"
  result_backend: "redis://:密码@主机:端口/数据库"
  default_routing_key: "w2t.capture"
  task_timeout_seconds: 100.0

capture:
  min_interval_ms: 1000  # 截图间隔（毫秒）
  enable_foreground_check: true

ocr:
  channels:
    - paddle_local
    - paddle_remote
    - tencent
  timeout_seconds: 5.0
```

## 使用方法

### 基本命令

#### 查看帮助信息

```bash
# 查看主命令帮助
python run.py --help

# 查看截图命令帮助
python run.py capture --help

# 查看处理命令帮助
python run.py process --help
```

### 截图命令

#### 使用默认目录开始截图

```bash
python run.py capture
```

#### 指定自定义保存目录

```bash
python run.py capture --dir my_screenshots
```

或使用短选项：

```bash
python run.py capture -d output
```

**使用说明**：
- 运行后会进入框选模式，用鼠标拖拽框选截图区域
- 右键点击确认选择，ESC 键取消
- 截图会按顺序保存为 `1.jpg`, `2.jpg`, `3.jpg`...
- 按 `Ctrl+C` 停止截图

### 处理图片命令

#### 直接处理模式（不使用 Celery）

```bash
# 处理默认目录 (results) 中的图片
python run.py process

# 处理指定目录中的图片
python run.py process --dir results

# 使用短选项
python run.py process -d my_screenshots
```

#### 使用 Celery 处理模式（并行处理）

**首先启动 Celery worker**（在另一个终端）：

```bash
celery -A src.worker worker --pool=eventlet --concurrency=4
```

**然后使用 Celery 模式处理图片**：

```bash
# 使用 Celery 处理默认目录
python run.py process --celery

# 指定目录并使用 Celery
python run.py process --dir results --celery

# 使用短选项
python run.py process -d output -c
```

## 完整工作流程示例

### 场景 1: 截图然后直接处理

```bash
# 步骤 1: 开始截图（按 Ctrl+C 停止）
python run.py capture

# 步骤 2: 处理截图（直接模式）
python run.py process
```

### 场景 2: 使用 Celery 并行处理大量图片

**终端 1**：启动 Celery worker（支持 4 个并发任务）
```bash
celery -A src.worker worker --pool=eventlet --concurrency=4
```

**终端 2**：开始截图
```bash
python run.py capture -d batch_screenshots
```

**终端 3**：使用 Celery 处理（会异步提交任务）
```bash
python run.py process -d batch_screenshots -c
```

### 场景 3: 处理已有图片目录

```bash
# 处理指定目录中已有的图片
python run.py process --dir /path/to/existing/images

# 或者使用 Celery 处理
python run.py process --dir /path/to/existing/images --celery
```

## 文件命名规则

- **图片文件**：`{序号}.jpg` (如 `1.jpg`, `2.jpg`, `3.jpg`)
- **文本文件**：`{序号}.txt` (如 `1.txt`, `2.txt`, `3.txt`)

文件按序号一一对应，方便追踪。

## 项目结构

```
W2T/
├── run.py              # 主入口脚本
├── settings.yaml       # 配置文件
├── requirements.txt    # Python 依赖
├── src/
│   ├── producer.py     # 截图生产者
│   ├── worker.py       # OCR Worker (Celery)
│   ├── screen_capture.py  # 屏幕截图模块
│   └── executor.py     # 动作执行器
└── results/            # 默认输出目录
```

## 注意事项

1. **截图命令**：
   - 框选模式中，右键确认选择，ESC 取消
   - 截图从 1 开始递增命名
   - 按 `Ctrl+C` 可随时停止

2. **处理命令**：
   - **直接模式**：顺序处理，速度较慢但简单，适合少量图片
   - **Celery 模式**：并行处理，速度快，适合大量图片，需要先启动 worker

3. **Celery Worker**：
   - 必须在处理图片之前启动
   - 可以根据需要调整并发数（`--concurrency`）
   - 使用 `eventlet` 池在 Windows 上运行

4. **配置文件**：
   - Redis 密码中的特殊字符需要进行 URL 编码（如 `@` → `%40`，`#` → `%23`）

## 常见问题

### Q: Celery worker 无法连接 Redis？

A: 检查 `settings.yaml` 中的 Redis 配置，确保密码已正确 URL 编码。

### Q: 截图很卡？

A: 可以调整 `settings.yaml` 中的 `min_interval_ms` 增加截图间隔。

### Q: OCR 识别速度慢？

A: 使用 Celery 模式可以并行处理多张图片，显著提升速度。

## License

MIT

