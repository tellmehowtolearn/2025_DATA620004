from sklearn.model_selection import ParameterGrid
from network import ThreeLayerFNN
from train import Trainer
from datetime import datetime
import numpy as np
import os
import pickle
import sys
from typing import Tuple
import time

def hyperparameter_search(X_train, y_train, X_val, y_val):
    # 定义超参数搜索空间
    param_grid = {
        'hidden_size1': [256, 512],
        'hidden_size2': [128, 256],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'reg_lambda': [0, 1e-5],
        'batch_size': [32, 64, 128],
        'activation': ['relu', 'tanh'],
    }

    # 使用 GridSearch 搜索所有超参数组合
    grid = ParameterGrid(param_grid)
    best_val_acc = 0
    best_params = {}

    for params in grid:
        print(f"Training with parameters: {params}")
        
        # 创建模型
        model = ThreeLayerFNN(input_size=X_train.shape[1], 
                              hidden_size1=params['hidden_size1'],
                              hidden_size2=params['hidden_size2'],
                              output_size=10,
                              activation=params['activation'],
                              reg_lambda=params['reg_lambda'])
        print("Model initialized.")
        print(f"input_size: {model.input_size}")
        print(f"hidden_size1: {model.hidden_size1}")
        print(f"hidden_size2: {model.hidden_size2}")
        print(f"activation: {model.activation_name}")
        print(f"reg_lambda: {model.reg_lambda}")

        # 创建训练器
        trainer = Trainer(model, X_train, y_train, X_val, y_val,
                          learning_rate=params['learning_rate'],
                          batch_size=params['batch_size'], 
                          epochs=10,
                          lr_decay=0.95, verbose=True)

        # 开始训练
        time_start = time.time()
        trainer.train()
        print(f"Training time: {time.time() - time_start:.2f} seconds")

        # 评估模型
        val_acc = trainer.evaluate(X_val, y_val)
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # 记录最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
        
        print(f"Best validation accuracy so far: {best_val_acc:.4f}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best hyperparameters: {best_params}")

    return best_params

def load_cifar10(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 dataset."""
    X_train = []
    y_train = []

    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            X_train.append(data_dict[b'data'])
            y_train.extend(data_dict[b'labels'])

    X_train = np.vstack(X_train).astype(np.float32)
    y_train = np.array(y_train)

    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        X_test = data_dict[b'data'].astype(np.float32)
        y_test = np.array(data_dict[b'labels'])

    # Normalize to [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    return X_train, y_train, X_test, y_test

def main():
    timestamp = datetime.now().strftime('%Y-%m-%d')
    log_file = open(f'hyperparameter_search_{timestamp}_log.txt', 'w')

    class Logger:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            if not self.log.closed:
                self.log.flush()

    sys.stdout = Logger(log_file)

    # 加载数据
    data_dir = './cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)

    # 分割训练集和验证集
    num_val = 5000
    X_val = X_train[:num_val]
    y_val = y_train[:num_val]
    X_train = X_train[num_val:]
    y_train = y_train[num_val:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # 超参数搜索
    best_params = hyperparameter_search(X_train, y_train, X_val, y_val)

    # 存储最优参数
    with open(f'best_params_{timestamp}.pkl', 'wb') as f:
        pickle.dump(best_params, f)

if __name__ == "__main__":
    main()
    