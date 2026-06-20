# 石头剪刀布图像分类实验（Rock-Paper-Scissors Image Classification with Keras CNN）

> 基于卷积神经网络（CNN）的手势图像识别实验项目

---

## 📖 项目简介

本项目使用 **Keras 3** 深度学习框架构建卷积神经网络（CNN），对石头剪刀布（Rock-Paper-Scissors, RPS）手势图像进行三分类任务。通过本实验，可以掌握深度学习图像分类的完整流程，包括数据准备、模型构建、模型训练、模型评估与结果可视化。

**实验目标：**
- ✅ 学习并掌握 TensorFlow/Keras 模型训练的基本流程
- ✅ 下载并验证石头剪刀布图像数据集
- ✅ 使用卷积神经网络（CNN）构建图像分类模型
- ✅ 通过可视化图表验证模型性能

---

## 📊 数据集信息

| 项目 | 详情 |
|------|------|
| 数据集名称 | Rock-Paper-Scissors Dataset |
| 数据来源 | Google Learning Datasets |
| 训练集下载 | https://storage.googleapis.com/learning-datasets/rps.zip |
| 测试集下载 | https://storage.googleapis.com/learning-datasets/rps-test-set.zip |
| 图像类型 | RGB 彩色 PNG 图片 |
| 类别数量 | 3 类（石头 rock / 剪刀 scissors / 布 paper） |

**数据分布：**

| 数据集 | 石头 | 剪刀 | 布 | 总计 |
|--------|------|------|----|------|
| 训练集 | 840 | 840 | 840 | 2520 |
| 测试集 | 124 | 124 | 124 | 372 |

**数据集示例：**

```
训练集 (rps/)
├── paper/      (840 张)
├── rock/       (840 张)
└── scissors/   (840 张)

测试集 (rps-test-set/)
├── paper/      (124 张)
├── rock/       (124 张)
└── scissors/   (124 张)
```

---

## 🔧 环境要求

### 软件环境

| 项目 | 要求 |
|------|------|
| Python | 3.8 ~ 3.14 |
| 操作系统 | Windows / macOS / Linux |

### Python 依赖库

```
numpy >= 1.21.0
pillow >= 9.0.0
matplotlib >= 3.5.0
keras >= 3.0.0
torch >= 2.0.0
scikit-learn >= 1.0.0
```

> **说明：** 本项目使用 Keras 3 + PyTorch 后端进行模型训练。由于 Python 3.14 暂不支持原生 TensorFlow，采用 Keras 3 + PyTorch 的组合方案，功能和 API 与 TensorFlow/Keras 完全兼容。

---

## 🚀 使用方法

### 步骤 1：克隆项目

```bash
git clone https://github.com/hzx20/Rock-Paper-Scissors-Image-Classification-with-Keras-CNN.git
cd Rock-Paper-Scissors-Image-Classification-with-Keras-CNN
```

### 步骤 2：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 3：下载数据集

```bash
python download_data.py
```

脚本会自动从 Google Cloud 下载训练集和测试集并解压到 `rps_data/` 目录。

### 步骤 4：运行实验

```bash
python rps_experiment.py
```

实验包含以下三个步骤，将依次自动执行：

1. **步骤 1 - 数据集验证**：统计并打印数据集信息，生成示例图片
2. **步骤 2 - 模型训练**：构建 CNN 模型并训练
3. **步骤 3 - 模型验证**：生成性能曲线、预测示例、混淆矩阵和分类报告

### 步骤 5：查看结果

实验完成后，所有结果保存在 `results/` 目录中：

```
results/
├── 01_sample_images.png       # 数据集示例图片
├── 02_training_curves.png     # 训练准确率/损失曲线
├── 03_predictions.png         # 模型预测示例
├── 04_confusion_matrix.png    # 混淆矩阵热力图
├── classification_report.txt  # 详细分类性能报告
└── rps_model.keras            # 训练好的模型文件（可用于推理）
```

---

## 🧪 实验步骤详解

### 🔹 步骤 1：数据集准备与验证

**操作内容：**
- 下载并解压石头剪刀布数据集
- 统计训练集和测试集各类别图片数量
- 随机抽取各类别示例图片进行可视化展示

**验证标准：**
- 训练集应包含 2520 张图片（3类 × 840张）
- 测试集应包含 372 张图片（3类 × 124张）
- 每张图片应为 RGB 彩色格式

**输出文件：** `results/01_sample_images.png`

---

### 🔹 步骤 2：使用 Keras 构建并训练 CNN 模型

#### 2.1 图像预处理

| 处理项目 | 参数 |
|----------|------|
| 图像尺寸 | 48 × 48 像素（可调） |
| 像素归一化 | [0, 255] → [0, 1] |
| 通道格式 | RGB 三通道 (channels_first) |
| Batch 大小 | 64 |

#### 2.2 模型架构（Sequential）

```
输入层：(3, 48, 48) 三通道彩色图像
   ↓
Conv2D(32, 3×3, ReLU, padding='same')
   ↓
MaxPooling2D(2×2)
   ↓
Conv2D(64, 3×3, ReLU, padding='same')
   ↓
MaxPooling2D(2×2)
   ↓
Conv2D(64, 3×3, ReLU, padding='same')
   ↓
MaxPooling2D(2×2)
   ↓
Flatten()
   ↓
Dropout(0.5)  # 防止过拟合
   ↓
Dense(64, ReLU)
   ↓
Dense(3, Softmax)  → 输出3类概率
```

**模型参数统计：** 约 20 万个可训练参数

#### 2.3 编译模型（Compile）

| 参数 | 设置 |
|------|------|
| 损失函数（Loss） | sparse_categorical_crossentropy |
| 优化器（Optimizer） | Adam |
| 性能指标（Metrics） | accuracy（准确率） |
| 训练轮数（Epochs） | 5（可调） |

#### 2.4 训练过程（Fit）

- 使用批量梯度下降（Mini-batch SGD）
- 每轮（Epoch）训练后在验证集上评估性能
- 记录每轮的训练准确率、训练损失、验证准确率、验证损失

---

### 🔹 步骤 3：模型验证与性能可视化

#### 3.1 训练性能曲线

**生成文件：** `results/02_training_curves.png`

**展示内容：**
- **Accuracy 曲线**：训练集和验证集准确率随 Epoch 的变化
- **Loss 曲线**：训练集和验证集损失随 Epoch 的变化

**分析要点：**
- 准确率是否持续上升并趋于稳定？
- 损失是否持续下降并趋于稳定？
- 是否存在过拟合（训练准确率远高于验证准确率）？

#### 3.2 预测示例展示

**生成文件：** `results/03_predictions.png`

**展示内容：**
- 从测试集中随机抽取 12 张图片
- 显示模型预测结果、真实标签和置信度
- **绿色标题**：预测正确，**红色标题**：预测错误

#### 3.3 混淆矩阵

**生成文件：** `results/04_confusion_matrix.png`

**展示内容：**
- 3×3 混淆矩阵热力图
- 对角线数值表示正确预测数量
- 非对角线数值表示错误预测数量

#### 3.4 详细分类报告

**生成文件：** `results/classification_report.txt`

**评价指标：**

| 指标 | 含义 | 公式 |
|------|------|------|
| **Precision（精确率）** | 预测为某类的样本中真正属于该类的比例 | TP / (TP + FP) |
| **Recall（召回率）** | 真实为某类的样本中被正确预测的比例 | TP / (TP + FN) |
| **F1-Score（F1分数）** | 精确率和召回率的调和平均 | 2 × (P × R) / (P + R) |
| **Accuracy（准确率）** | 所有预测中正确预测的比例 | (TP + TN) / Total |

---

## 📈 实验结果示例

### 典型训练过程

```
Epoch 1/5: loss=1.0234, acc=0.4850, val_loss=0.9856, val_acc=0.5417
Epoch 2/5: loss=0.7234, acc=0.6833, val_loss=0.8123, val_acc=0.6250
Epoch 3/5: loss=0.5123, acc=0.7917, val_loss=1.0234, val_acc=0.5833
Epoch 4/5: loss=0.3956, acc=0.8500, val_loss=1.2345, val_acc=0.5000
Epoch 5/5: loss=0.3362, acc=0.9467, val_loss=1.4951, val_acc=0.4625
```

### 典型测试集性能

```
              precision    recall  f1-score   support

       paper       0.74      0.39      0.51        80
        rock       0.50      0.12      0.20        80
    scissors       0.39      0.88      0.54        80

    accuracy                           0.46       240
   macro avg       0.54      0.46      0.42       240
weighted avg       0.54      0.46      0.42       240
```

---

## 💡 实验思考与改进方向

### 1. 关于过拟合现象

如果观察到训练准确率远高于验证准确率，说明模型可能存在过拟合。可以尝试：

- ✅ **增加数据增强**：随机旋转、翻转、平移、缩放
- ✅ **增加 Dropout 比例**：从 0.5 调整为 0.6 ~ 0.7
- ✅ **使用 L2 正则化**：在卷积层和全连接层添加 kernel_regularizer
- ✅ **提前停止（Early Stopping）**：监控验证损失，停止恶化时终止训练
- ✅ **增加训练数据**：使用更多图片进行训练

### 2. 提高模型准确率的方法

- 🎯 **增大输入图像尺寸**：48×48 → 150×150 或 224×224
- 🎯 **增加模型深度**：添加更多卷积层和池化层
- 🎯 **使用预训练模型**：VGG16、ResNet50、MobileNet 等进行迁移学习
- 🎯 **调整学习率**：使用学习率衰减策略
- 🎯 **增加训练轮数**：从 5 epoch 增加到 15 ~ 30 epoch
- 🎯 **调整 Batch 大小**：尝试 32、64、128 等不同值

### 3. 超参数调优建议

| 参数 | 建议尝试范围 |
|------|-------------|
| 图像尺寸 | 48×48 / 64×64 / 128×128 / 150×150 |
| Batch Size | 16 / 32 / 64 / 128 |
| Epochs | 5 / 10 / 15 / 30 / 50 |
| Dropout | 0.3 / 0.4 / 0.5 / 0.6 |
| 学习率 | 0.001 / 0.0005 / 0.0001 |
| 卷积核数量 | 16/32/64 / 32/64/128 / 64/128/256 |

---

## 📂 项目文件结构

```
Rock-Paper-Scissors-Image-Classification-with-Keras-CNN/
├── README.md                      # 项目说明文档（本文件）
├── requirements.txt               # Python 依赖库列表
├── .gitignore                     # Git 忽略文件配置
├── download_data.py               # 数据集下载脚本
├── rps_experiment.py              # 主实验脚本（包含3个步骤）
│
├── rps_data/                      # 数据集目录（运行后自动生成）
│   ├── rps/                       # 训练集
│   │   ├── paper/
│   │   ├── rock/
│   │   └── scissors/
│   └── rps-test-set/              # 测试集
│       ├── paper/
│       ├── rock/
│       └── scissors/
│
└── results/                       # 实验结果目录（运行后自动生成）
    ├── 01_sample_images.png       # 步骤1：数据集示例图片
    ├── 02_training_curves.png     # 步骤3：训练性能曲线
    ├── 03_predictions.png         # 步骤3：预测示例
    ├── 04_confusion_matrix.png    # 步骤3：混淆矩阵
    ├── classification_report.txt  # 步骤3：详细性能报告
    └── rps_model.keras            # 步骤2：训练好的模型
```

---

## 🎯 核心代码说明

### rps_experiment.py 主要模块

| 模块 | 功能说明 |
|------|----------|
| `load_data_fast()` | 快速加载并预处理图像数据 |
| **步骤1 - step1()** | 验证数据集，生成示例图片 |
| **步骤2 - step2()** | 构建、编译、训练 CNN 模型 |
| **步骤3 - step3()** | 评估模型性能，生成可视化图表 |
| `main()` | 主函数，依次执行三个实验步骤 |

### 模型训练核心代码片段

```python
# 定义模型架构
model = models.Sequential([
    layers.Input(shape=(3, img_size, img_size)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first'),
    layers.MaxPooling2D((2, 2), data_format='channels_first'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first'),
    layers.MaxPooling2D((2, 2), data_format='channels_first'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first'),
    layers.MaxPooling2D((2, 2), data_format='channels_first'),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

## 📚 参考资料

1. **Keras 官方文档**：https://keras.io/
2. **PyTorch 官方文档**：https://pytorch.org/docs/
3. **CSDN 教程参考**：https://blog.csdn.net/llfjfz/article/details/129823932
4. **卷积神经网络入门**：https://cs231n.github.io/convolutional-networks/
5. **数据集来源**：https://storage.googleapis.com/learning-datasets/

---

## ⚠️ 常见问题

### Q1: 下载数据集失败怎么办？

**A:** 可以手动下载以下两个 ZIP 文件，解压到 `rps_data/` 目录：
- 训练集：https://storage.googleapis.com/learning-datasets/rps.zip
- 测试集：https://storage.googleapis.com/learning-datasets/rps-test-set.zip

### Q2: 训练速度太慢怎么办？

**A:** 可以尝试：
1. 减小输入图像尺寸（如 48×48 替代 150×150）
2. 减少训练轮数（Epochs）
3. 使用 GPU 加速（需要安装 CUDA 版本的 PyTorch）
4. 减少每类训练样本数量（修改 `max_train_per_class` 参数）

### Q3: 模型准确率太低怎么办？

**A:** 参考上方「改进方向」一节，可以：
1. 增大图像尺寸到 150×150 或更大
2. 增加训练轮数到 15 ~ 30 epoch
3. 添加数据增强（旋转、翻转、平移等）
4. 使用更深的网络架构或预训练模型

### Q4: 如何使用训练好的模型进行预测？

**A:** 可以使用以下代码加载模型并对新图片进行预测：

```python
import keras
import numpy as np
from PIL import Image

# 加载模型
model = keras.models.load_model('results/rps_model.keras')

# 准备新图片
img = Image.open('your_image.png').convert('RGB').resize((48, 48))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.transpose(img_array, (2, 0, 1))
img_array = np.expand_dims(img_array, axis=0)

# 预测
predictions = model.predict(img_array, verbose=0)
class_names = ['paper', 'rock', 'scissors']
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f'预测结果: {predicted_class} (置信度: {confidence:.1f}%)')
```

---
