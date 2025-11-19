"""
基于深度学习的图片分类模块。
将图片10和21分类为一类，其余图片分类为另一类。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 尝试导入深度学习框架
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，可以使用: pip install torch torchvision")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: sklearn未安装，可以使用: pip install scikit-learn")


def extract_number(filename: str) -> int:
    """从文件名中提取数字用于排序"""
    import re
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0


class ImageDataset(Dataset):
    """图片数据集类"""
    
    def __init__(self, images: List[np.ndarray], labels: List[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 转换为RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL Image格式（PyTorch需要）
        from PIL import Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class SimpleCNN(nn.Module):
    """简单的CNN分类模型"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 自适应池化层（自动计算尺寸）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 卷积层1
        x = self.pool(self.relu(self.conv1(x)))  # 256x256 -> 128x128
        # 卷积层2
        x = self.pool(self.relu(self.conv2(x)))  # 128x128 -> 64x64
        # 卷积层3
        x = self.pool(self.relu(self.conv3(x)))  # 64x64 -> 32x32
        
        # 自适应池化到固定尺寸
        x = self.adaptive_pool(x)  # 32x32 -> 8x8
        
        # 展平
        x = x.view(-1, 128 * 8 * 8)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def load_images_and_labels(results_dir: str | Path = "results") -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    加载图片和标签。
    
    标签规则：
    - 图片10和21: 标签为1（一类）
    - 其他图片: 标签为0（另一类）
    
    Args:
        results_dir: 存放图片的目录路径
    
    Returns:
        (图片列表, 标签列表, 文件名列表)
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"目录不存在: {results_dir}")
    
    # 获取所有jpg图片文件并排序
    image_files = sorted(
        [f for f in os.listdir(results_path) if f.lower().endswith((".jpg", ".jpeg"))],
        key=extract_number
    )
    
    images = []
    labels = []
    filenames = []
    
    print(f"找到 {len(image_files)} 张图片，开始加载...")
    
    for img_file in image_files:
        img_path = results_path / img_file
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"警告: 无法读取图片 {img_file}")
            continue
        
        # 统一尺寸为256x256
        img = cv2.resize(img, (256, 256))
        images.append(img)
        filenames.append(img_file)
        
        # 根据文件名确定标签
        img_num = extract_number(img_file)
        if img_num in [13,24,35,45,55,66,77,87,98,109,120]:
            labels.append(1)  # 一类（10和21）
            print(f"  {img_file}: 标签=1 (类别1)")
        else:
            labels.append(0)  # 另一类（其他）
            print(f"  {img_file}: 标签=0 (类别0)")
    
    print(f"\n加载完成！")
    print(f"  类别0（其他）: {labels.count(0)} 张")
    print(f"  类别1（10和21）: {labels.count(1)} 张")
    
    return images, labels, filenames


def train_model(images: List[np.ndarray], labels: List[int], 
                epochs: int = 50, batch_size: int = 4, 
                learning_rate: float = 0.001) -> nn.Module:
    """
    训练深度学习分类模型。
    
    Args:
        images: 图片列表
        labels: 标签列表
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    
    Returns:
        训练好的模型
    """
    if not TORCH_AVAILABLE:
        raise ImportError("此功能需要安装 PyTorch: pip install torch torchvision")
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("此功能需要安装 sklearn: pip install scikit-learn")
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 张")
    print(f"  验证集: {len(X_val)} 张")
    
    # 创建数据集
    train_dataset = ImageDataset(X_train, y_train, transform=transform_train)
    val_dataset = ImageDataset(X_val, y_val, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    print(f"\n开始训练（{epochs}轮）...")
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for images_batch, labels_batch in train_loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images_batch, labels_batch in val_loader:
                images_batch = images_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(images_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"轮次 {epoch+1}/{epochs}: 训练损失={train_loss:.4f}, 验证准确率={val_acc:.2f}%")
    
    print(f"\n训练完成！最终验证准确率: {val_accuracies[-1]:.2f}%")
    
    # 绘制训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses)
    axes[0].set_title('训练损失')
    axes[0].set_xlabel('轮次')
    axes[0].set_ylabel('损失')
    axes[0].grid(True)
    
    axes[1].plot(val_accuracies)
    axes[1].set_title('验证准确率')
    axes[1].set_xlabel('轮次')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model


def predict_all_images(model: nn.Module, images: List[np.ndarray], 
                      filenames: List[str], device: torch.device) -> List[Tuple[str, int, float]]:
    """
    对所有图片进行预测。
    
    Args:
        model: 训练好的模型
        images: 图片列表
        filenames: 文件名列表
        device: 设备（CPU或GPU）
    
    Returns:
        [(文件名, 预测类别, 置信度), ...]
    """
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    with torch.no_grad():
        for img, filename in zip(images, filenames):
            # 转换为RGB
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL Image
            from PIL import Image
            img_pil = Image.fromarray(img_rgb)
            
            # 预处理
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # 预测
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            results.append((filename, predicted_class, confidence))
    
    return results


def visualize_predictions(results: List[Tuple[str, int, float]], 
                         true_labels: List[int],
                         filenames: List[str]) -> None:
    """
    可视化预测结果。
    
    Args:
        results: 预测结果列表 [(文件名, 预测类别, 置信度), ...]
        true_labels: 真实标签列表
        filenames: 文件名列表
    """
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：预测结果柱状图
    ax1 = axes[0]
    pred_classes = [r[1] for r in results]
    confidences = [r[2] for r in results]
    img_names = [Path(r[0]).stem for r in results]
    
    colors = ['red' if pred == 1 else 'blue' for pred in pred_classes]
    bars = ax1.bar(range(len(results)), confidences, color=colors, alpha=0.7)
    
    # 标记真实标签
    for i, (pred, true) in enumerate(zip(pred_classes, true_labels)):
        if pred != true:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(3)
            bars[i].set_alpha(0.5)
    
    ax1.set_xlabel('图片', fontsize=12)
    ax1.set_ylabel('置信度', fontsize=12)
    ax1.set_title('预测结果（红色=类别1(10和21)，蓝色=类别0(其他)）', fontsize=14)
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(img_names, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='阈值')
    ax1.legend()
    
    # 下图：混淆矩阵
    ax2 = axes[1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_classes)
    
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.figure.colorbar(im, ax=ax2)
    
    # 添加文本标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax2.set_ylabel('真实标签', fontsize=12)
    ax2.set_xlabel('预测标签', fontsize=12)
    ax2.set_title('混淆矩阵', fontsize=14)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['类别0', '类别1'])
    ax2.set_yticklabels(['类别0', '类别1'])
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细结果
    print("\n" + "=" * 60)
    print("预测结果详情")
    print("=" * 60)
    correct = 0
    for (filename, pred_class, confidence), true_label in zip(results, true_labels):
        status = "✓" if pred_class == true_label else "✗"
        if pred_class == true_label:
            correct += 1
        class_name = "类别1(10和21)" if pred_class == 1 else "类别0(其他)"
        print(f"{status} {filename}: 预测={class_name}, 置信度={confidence:.4f}, 真实={'类别1' if true_label == 1 else '类别0'}")
    
    print(f"\n总准确率: {correct}/{len(results)} = {100*correct/len(results):.2f}%")
    
    # 打印分类报告
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    from sklearn.metrics import classification_report
    print(classification_report(true_labels, pred_classes, 
                                target_names=['类别0(其他)', '类别1(10和21)']))


def load_trained_model(model_path: str | Path = "results/classifier_model.pth", 
                       device: Optional[torch.device] = None) -> nn.Module:
    """
    加载已训练的模型。
    
    Args:
        model_path: 模型文件路径
        device: 设备（CPU或GPU），如果为None则自动选择
    
    Returns:
        加载好的模型
    """
    if not TORCH_AVAILABLE:
        raise ImportError("此功能需要安装 PyTorch: pip install torch torchvision")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # print(f"从 {model_path} 加载模型...")
    # print(f"使用设备: {device}")
    
    # 创建模型结构
    model = SimpleCNN(num_classes=2)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # print("模型加载完成！")
    return model


def predict_single_image(model: nn.Module, 
                        image_path: str | Path,
                        device: Optional[torch.device] = None) -> Tuple[int, float, str]:
    """
    预测单张图片的类别。
    
    Args:
        model: 训练好的模型
        image_path: 图片文件路径
        device: 设备（CPU或GPU），如果为None则自动选择
    
    Returns:
        (预测类别, 置信度, 类别名称)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 统一尺寸
    img = cv2.resize(img, (256, 256))
    
    # 转换为RGB
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    from PIL import Image
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_name = "类别1(10和21)" if predicted_class == 1 else "类别0(其他)"
    
    return predicted_class, confidence, class_name


def predict_images_from_dir(model: nn.Module,
                           image_dir: str | Path,
                           device: Optional[torch.device] = None) -> List[Tuple[str, int, float, str]]:
    """
    预测目录中所有图片的类别。
    
    Args:
        model: 训练好的模型
        image_dir: 图片目录路径
        device: 设备（CPU或GPU），如果为None则自动选择
    
    Returns:
        [(文件名, 预测类别, 置信度, 类别名称), ...]
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"目录不存在: {image_dir}")
    
    # 获取所有图片文件
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=extract_number
    )
    
    if not image_files:
        print(f"目录中没有找到图片文件")
        return []
    
    print(f"找到 {len(image_files)} 张图片，开始预测...")
    
    results = []
    for img_file in image_files:
        try:
            img_path = image_dir / img_file
            pred_class, confidence, class_name = predict_single_image(model, img_path, device)
            results.append((img_file, pred_class, confidence, class_name))
            print(f"  {img_file}: {class_name}, 置信度={confidence:.4f}")
        except Exception as e:
            print(f"  警告: 预测 {img_file} 时出错: {e}")
            continue
    
    return results


def main():
    """主函数"""
    import sys
    
    if not TORCH_AVAILABLE:
        print("错误: 需要安装 PyTorch")
        print("请运行: pip install torch torchvision")
        return
    
    if not SKLEARN_AVAILABLE:
        print("错误: 需要安装 sklearn")
        print("请运行: pip install scikit-learn")
        return
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "predict":
            # 预测模式
            if len(sys.argv) < 3:
                print("使用方法: python deep_classifier.py predict <图片路径或目录>")
                print("示例:")
                print("  python deep_classifier.py predict results/1.jpg")
                print("  python deep_classifier.py predict results/")
                return
            
            target_path = Path(sys.argv[2])
            model_path = Path("results") / "classifier_model.pth"
            
            if not model_path.exists():
                print(f"错误: 模型文件不存在: {model_path}")
                print("请先运行训练: python deep_classifier.py train")
                return
            
            # 加载模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_trained_model(model_path, device)
            
            # 预测
            if target_path.is_file():
                # 单张图片
                print(f"\n预测图片: {target_path}")
                pred_class, confidence, class_name = predict_single_image(model, target_path, device)
                print(f"\n预测结果:")
                print(f"  类别: {class_name}")
                print(f"  置信度: {confidence:.4f}")
            elif target_path.is_dir():
                # 目录中的所有图片
                print(f"\n预测目录: {target_path}")
                results = predict_images_from_dir(model, target_path, device)
                
                # 统计结果
                class0_count = sum(1 for _, pred, _, _ in results if pred == 0)
                class1_count = sum(1 for _, pred, _, _ in results if pred == 1)
                
                print(f"\n预测结果统计:")
                print(f"  类别0(其他): {class0_count} 张")
                print(f"  类别1(10和21): {class1_count} 张")
            else:
                print(f"错误: 路径不存在: {target_path}")
            return
        
        elif command == "train":
            # 训练模式
            print("=" * 60)
            print("基于深度学习的图片分类 - 训练模式")
            print("=" * 60)
            print("分类规则:")
            print("  - 类别1: 图片10和21")
            print("  - 类别0: 其他所有图片")
            print("=" * 60)
            
            # 1. 加载数据
            images, labels, filenames = load_images_and_labels("results/train")
            
            if len(images) < 4:
                print("错误: 图片数量太少，无法训练")
                return
            
            # 2. 训练模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = train_model(images, labels, epochs=50, batch_size=4, learning_rate=0.001)
            
            # 3. 对所有图片进行预测
            print("\n对所有图片进行预测...")
            results = predict_all_images(model, images, filenames, device)
            
            # 4. 可视化结果
            visualize_predictions(results, labels, filenames)
            
            # 5. 保存模型
            model_path = Path("results") / "classifier_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"\n模型已保存到: {model_path}")
            return
    
    # 默认：训练模式
    print("=" * 60)
    print("基于深度学习的图片分类")
    print("=" * 60)
    print("分类规则:")
    print("  - 类别1: 图片10和21")
    print("  - 类别0: 其他所有图片")
    print("=" * 60)
    print("\n使用方法:")
    print("  训练模型: python deep_classifier.py train")
    print("  预测图片: python deep_classifier.py predict <图片路径或目录>")
    print("=" * 60)
    
    # 1. 加载数据
    images, labels, filenames = load_images_and_labels("results")
    
    if len(images) < 4:
        print("错误: 图片数量太少，无法训练")
        return
    
    # 2. 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(images, labels, epochs=50, batch_size=4, learning_rate=0.001)
    
    # 3. 对所有图片进行预测
    print("\n对所有图片进行预测...")
    results = predict_all_images(model, images, filenames, device)
    
    # 4. 可视化结果
    visualize_predictions(results, labels, filenames)
    
    # 5. 保存模型
    model_path = Path("results") / "classifier_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")


if __name__ == "__main__":
    main()

