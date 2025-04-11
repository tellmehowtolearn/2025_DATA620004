import numpy as np

class ThreeLayerFNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=10, activation='relu', reg_lambda=0.01):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.activation_name = activation
        self.reg_lambda = reg_lambda

        # 初始化权重和偏置（使用高斯分布初始化）
        self.params = {
            'W1': np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size),
            'b1': np.zeros((1, hidden_size1)),
            'W2': np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1),
            'b2': np.zeros((1, hidden_size2)),
            'W3': np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2),
            'b3': np.zeros((1, output_size)),
        }

    def activation(self, z):
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, z):
        if self.activation_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        else:
            raise ValueError("Unsupported activation function")

    def softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))  # 稳定性优化
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        core_loss = -np.sum(np.log(y_pred[range(m), y_true])) / m
        # 加上 L2 正则项
        reg_loss = 0.5 * self.reg_lambda * (
            np.sum(np.square(self.params['W1'])) +
            np.sum(np.square(self.params['W2'])) +
            np.sum(np.square(self.params['W3']))
        )
        return core_loss + reg_loss

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        z1 = X.dot(W1) + b1
        a1 = self.activation(z1)
        z2 = a1.dot(W2) + b2
        a2 = self.activation(z2)
        z3 = a2.dot(W3) + b3
        a3 = self.softmax(z3)

        cache = {
            'X': X, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3
        }
        return a3, cache

    def backward(self, y_true, cache):
        m = y_true.shape[0]
        grads = {}
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        a1, a2, a3 = cache['a1'], cache['a2'], cache['a3']
        z1, z2 = cache['z1'], cache['z2']
        X = cache['X']

        # One-hot 编码
        y_one_hot = np.zeros_like(a3)
        y_one_hot[np.arange(m), y_true] = 1

        # 输出层梯度
        dz3 = (a3 - y_one_hot) / m
        grads['W3'] = a2.T.dot(dz3) + self.reg_lambda * W3
        grads['b3'] = np.sum(dz3, axis=0, keepdims=True)

        # 第二层梯度
        da2 = dz3.dot(W3.T)
        dz2 = da2 * self.activation_derivative(z2)
        grads['W2'] = a1.T.dot(dz2) + self.reg_lambda * W2
        grads['b2'] = np.sum(dz2, axis=0, keepdims=True)

        # 第一层梯度
        da1 = dz2.dot(W2.T)
        dz1 = da1 * self.activation_derivative(z1)
        grads['W1'] = X.T.dot(dz1) + self.reg_lambda * W1
        grads['b1'] = np.sum(dz1, axis=0, keepdims=True)

        return grads
    
    def update_params(self, grads, learning_rate):
        for key in self.params:
            self.params[key] -= learning_rate * grads[key]

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)
