import numpy as np
import os
import pickle
from typing import Tuple
import sys
from network import ThreeLayerFNN
from datetime import datetime
import copy
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, 
                 learning_rate=1e-3, batch_size=64, 
                 epochs=100, lr_decay=0.95, verbose=True):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.verbose = verbose

        # Initialize the history lists for loss and accuracy
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        # For saving best model
        self.best_val_acc = 0.0
        self.best_model_params = None



    def train(self):
        num_train = self.X_train.shape[0]
        for epoch in range(1, self.epochs + 1):
            # Shuffle training data
            indices = np.arange(num_train)
            np.random.shuffle(indices)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]

            # Mini-batch training
            for i in range(0, num_train, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                y_pred, cache = self.model.forward(X_batch)
                loss = self.model.compute_loss(y_batch, y_pred)
                grads = self.model.backward(y_batch, cache)
                self.model.update_params(grads, self.learning_rate)

            y_train_pred, _ = self.model.forward(self.X_train)
            self.train_loss_history.append(self.model.compute_loss(self.y_train, y_train_pred))
            train_preds = self.model.predict(self.X_train)
            val_preds = self.model.predict(self.X_val)
            y_val_pred, _ = self.model.forward(self.X_val)
            self.val_loss_history.append(self.model.compute_loss(self.y_val, y_val_pred))
            train_acc = np.mean(train_preds == self.y_train)
            val_acc = np.mean(val_preds == self.y_val)
            self.val_acc_history.append(val_acc)

            # Decay learning rate
            self.learning_rate *= self.lr_decay

            # 保存最优模型参数
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_params = copy.deepcopy(self.model.params)
                if self.verbose:
                    print(f">>> New best model saved with val_acc = {val_acc:.4f}")

            if self.verbose and (epoch % 1 == 0 or epoch == 1 or epoch == self.epochs):
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Train Acc = {train_acc:.2%}, Val Acc = {val_acc:.2%}")
    
    def evaluate(self, X, y):
        preds = self.model.predict(X)
        return np.mean(preds == y)
    
    def plot_history(self, save_path=None):
        plt.figure(figsize=(12, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.plot(self.val_loss_history, label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.val_acc_history, label='Val Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()

        # Save plot if save_path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()

    def save_best_model(self, file_path):
        if self.best_model_params is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.best_model_params, f)
            print(f"Best model parameters saved to {file_path}")
        else:
            print("No best model to save.")

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
    log_file = open(f'train_{timestamp}_log.txt', 'w')

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

    data_dir = './cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Split validation set
    num_val = 5000
    X_val = X_train[:num_val]
    y_val = y_train[:num_val]
    X_train = X_train[num_val:]
    y_train = y_train[num_val:]

    # 模型初始化
    input_size = X_train.shape[1]
    model = ThreeLayerFNN(input_size=input_size, hidden_size1=512, hidden_size2=256, 
                         output_size=10, activation='relu', reg_lambda=0)
    print("Model initialized.")
    print(f"input_size: {model.input_size}")
    print(f"hidden_size1: {model.hidden_size1}")
    print(f"hidden_size2: {model.hidden_size2}")
    print(f"activation: {model.activation_name}")
    print(f"reg_lambda: {model.reg_lambda}")

    trainer = Trainer(model, X_train, y_train, X_val, y_val, 
                      learning_rate=1e-2, batch_size=64, 
                      epochs=100, lr_decay=0.97, verbose=True)

    print("Initial validation accuracy:", trainer.evaluate(X_val, y_val))
    trainer.train()
    trainer.plot_history()  # <--- 绘图展示

    val_acc = trainer.evaluate(X_test, y_test)
    print(f"Test accuracy: {val_acc:.4f}")

    # Save training history
    history_file = f'trainer_history_h1_{model.hidden_size1}_h2_{model.hidden_size2}_lr_{trainer.learning_rate:.0e}_reg_{model.reg_lambda}.pkl'
    with open(history_file, 'wb') as f:
        pickle.dump({
            'train_loss_history': trainer.train_loss_history,
            'val_loss_history': trainer.val_loss_history,
            'val_acc_history': trainer.val_acc_history
        }, f)

    # Save model parameters
    model_params = {
        'weights': model.params,
        'input_size': model.input_size,
        'hidden_size1': model.hidden_size1,
        'hidden_size2': model.hidden_size2,
        'activation': model.activation_name,
        'reg_lambda': model.reg_lambda
    }
    model_file = f'model_parameters_h1_{model.hidden_size1}_h2_{model.hidden_size2}_lr_{trainer.learning_rate}_reg_{model.reg_lambda}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_params, f)

    print("Model parameters saved.")

if __name__ == '__main__':
    main()
