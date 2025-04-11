import os
import pickle
import sys
import numpy as np
from datetime import datetime

from network import ThreeLayerFNN
from train import Trainer, load_cifar10

def load_best_params(param_dir='.'):
    """Load the most recent best_params_<timestamp>.pkl file."""
    best_file = sorted([f for f in os.listdir(param_dir) if f.startswith('best_params_') and f.endswith('.pkl')])[-1]
    with open(os.path.join(param_dir, best_file), 'rb') as f:
        best_params = pickle.load(f)
    print(f"Loaded best parameters from {best_file}")
    return best_params

def main():
    # Redirect logs
    timestamp = datetime.now().strftime('%Y-%m-%d')
    log_file = open(f'test_log_{timestamp}.txt', 'w')
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

    # Load CIFAR-10
    data_dir = './cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)

    # Validation split
    num_val = 5000
    X_val = X_train[:num_val]
    y_val = y_train[:num_val]
    X_train = X_train[num_val:]
    y_train = y_train[num_val:]

    # Load best hyperparameters
    best_params = load_best_params()
    print(f"Best parameters: {best_params}")

    model = ThreeLayerFNN(
        input_size=X_train.shape[1],
        hidden_size1=best_params['hidden_size1'],
        hidden_size2=best_params['hidden_size2'],
        output_size=10,
        activation=best_params['activation'],
        reg_lambda=best_params['reg_lambda']
    )

    trainer = Trainer(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        epochs=100,  # increase if needed
        lr_decay=0.95,
        verbose=True
    )

    # Check if pretrained weights exist
    model_filename = f"best_model_{best_params['hidden_size1']}_{best_params['hidden_size2']}_{best_params['learning_rate']}_{best_params['reg_lambda']}_{best_params['activation']}.pkl"
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            model.params = pickle.load(f)
        print(f"Loaded trained model weights from {model_filename}")
    else:
        print("No pretrained model found, training from scratch.")
        trainer.train()
        fig_name = f"train_history_{best_params['hidden_size1']}_{best_params['hidden_size2']}_{best_params['learning_rate']}_{best_params['reg_lambda']}_{best_params['activation']}.png"
        trainer.plot_history(save_path=fig_name)
        trainer.save_best_model(model_filename)

    test_acc = trainer.evaluate(X_test, y_test)
    print(f"Test set accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
