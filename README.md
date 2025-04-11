# 2025_DATA620004
2025春季学期神经网络和深度学习第一次作业（从零开始构建三层神经网络分类器，实现图像分类）模型训练与测试说明文档

## 1. 参数查找：多超参数组合的网格搜索

本项目通过网格搜索方式对以下六个关键超参数进行了系统组合评估：

- 第一层隐藏层大小：`[256, 512]`
- 第二层隐藏层大小：`[128, 256]`
- 学习率：`[1e-4, 1e-3, 1e-2]`
- 正则化强度（L2）：`[0, 1e-5]`
- 批量大小（batch_size）：`[32, 64, 128]`
- 激活函数：`['relu', 'tanh']`

总共组合数为 $2 \times 2 \times 3 \times 2 \times 3 \times 2 = 144$ 种。所有组合均使用相同训练轮数（10轮）进行训练，并在验证集上评估性能。最终选取验证集损失最小且准确率最高的超参数组合作为最终模型配置。

超参数搜索逻辑实现于 [`hyperparameter_search.py`](./hyperparameter_search.py)，最优参数将自动保存为 `best_params_<timestamp>.pkl` 文件。

---

## 2. 模型训练说明

模型训练流程基于 `train.py` 中定义的 `Trainer` 类与`network.py`中定义的 `ThreeLayerFNN` 类，支持完整训练、日志记录、模型保存与绘图。

### 2.1 激活函数

测试`ReLU` 与 `Tanh` 两种激活函数通过，可由参数动态指定，默认在超参数搜索中进行选择。

### 2.2 反向传播、损失与梯度计算

- 损失函数采用交叉熵加 L2 正则化形式（定义在 `compute_loss` 函数中）。
- 梯度计算依照链式法则进行（Downstream Gradient = Local Gradient × Upstream Gradient），最终对权重矩阵 $W_1$, $W_2$, $W_3$, $b_1$,$b_2$,$b_3$ 进行梯度更新，逻辑封装在 `backward` 方法中。
- softmax 对数损失的梯度简化为 `softmax - y_truth`，可直接应用。

### 2.3 学习率下降策略

每个epoch之后，学习率按照指数衰减策略下降（`lr *= lr_decay`），其中 `lr_decay = 0.95`。

### 2.4 L2 正则化

正则化项定义为：

$$
\frac{\lambda}{2} \|W_1\|_2^2 + \frac{\lambda}{2} \|W_2\|_2^2 + \frac{\lambda}{2} \|W_3\|_2^2
$$

最终梯度需加上 $\lambda W_1$ 、$\lambda W_2$、$\lambda W_3$  项。

### 2.5 SGD 优化器

采用小批量梯度下降（SGD），每次训练选取 `batch_size` 大小的样本子集进行前向传播和反向传播，提升训练效率。

### 2.6 模型保存与训练历史

- 训练过程中自动保存模型权重（`w1`, `b1`, `w2`, `b2`, `w3`, `b3`）至trainer.best_model_params，如果最终需要保存为文件，就使用`save_best_model()`方法。
- 同时记录训练与验证过程中的损失与准确率，并能够绘图输出。

---

### 最终模型训练

最终模型训练阶段基于参数搜索过程中保存的最优超参数文件`best_params_<timestamp>.pkl`，自动加载包括学习率、隐藏层规模、正则化强度、激活函数类型、批量大小等配置，并在进行更大轮次的训练以充分拟合模型。训练过程中不仅自动记录训练集与验证集上的损失和准确率，还会保存每轮训练历史，并生成可视化曲线图以便分析模型性能变化。同时引入最优验证集准确率追踪机制，确保自动保存性能最优时刻的模型参数。
整体训练与日志管理逻辑由  [`test.py`](./test.py) 中的 `main()` 函数统一控制，实现包括：标准输出与日志文件同步、历史记录、最佳模型权重存储等功能。
注意训练完成后，仍可使用由 [`test.py`](./test.py) 中的 `main()`加载已保存模型权重，结合最优超参数文件进行测试集评估，输出分类准确率（Accuracy），形成完整训练-验证-测试闭环，便于后续模型部署与对比分析。


---

## 3. 测试流程

测试过程由 `test.py` 脚本完成，主要功能包括：

- 自动加载最优参数配置（从 `best_params_*.pkl` 中读取）
- 加载训练完成的模型权重
- 在测试集上进行前向推理，输出准确率（Accuracy）

---

## 4. 使用流程（推荐顺序）

```bash
# 第一步：进行超参数搜索（含模型初步训练，可以在文件里进行超参数选择的范围调整）
python hyperparameter_search.py


# 第二步：基于最优参数进行最终训练以及模型评估（在测试集上评估准确率）
python test.py
