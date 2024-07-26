import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from matplotlib.animation import FuncAnimation

# -------------------- 데이터 불러오기 및 전처리 --------------------

# MNIST 데이터셋 로드
mnist = fetch_openml('mnist_784', version=1)
X = mnist['data'].values  # numpy 배열로 변환
y = mnist['target'].astype(int).values  # numpy 배열로 변환

# 데이터 전처리
X = X / 255.0
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

# 훈련셋과 테스트셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# -------------------- 활성화 함수 정의 --------------------
class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.output * (1 - self.output))

class ReLU:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.output <= 0] = 0
        return grad_input

# -------------------- 로스 클래스 정의 --------------------
class BinaryCrossEntropyLoss:
    def __init__(self):
        self.grad = None

    def __call__(self, y_pred, y_true):
        epsilon = 1e-5
        self.y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        self.y_true = y_true
        loss = -np.mean(y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        return loss

    def backward(self):
        self.grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * len(self.y_true))
        return self.grad

# -------------------- 옵티마이저 클래스 정의 --------------------
class SGDOptimizer:
    def __init__(self, params, learning_rate=0.01):
        self.params = params
        self.learning_rate = learning_rate

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.grad)

    def step(self):
        for param in self.params:
            param.data -= self.learning_rate * param.grad

# -------------------- 커스텀 MLP 모델 클래스 정의 --------------------
class CustomMLP:
    def __init__(self, input_dim, hidden_layers, output_dim):
        self.layers = []
        prev_dim = input_dim

        if hidden_layers == [0, 0]:
            self.layers.append(DenseLayer(prev_dim, output_dim))
            self.layers.append(Sigmoid())  # Output layer activation function
        else:
            for hidden_dim in hidden_layers:
                if hidden_dim > 0:
                    self.layers.append(DenseLayer(prev_dim, hidden_dim))
                    self.layers.append(ReLU())
                    prev_dim = hidden_dim

            self.layers.append(DenseLayer(prev_dim, output_dim))
            self.layers.append(Sigmoid())  # Output layer activation function

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                params.append(layer.weight)
                params.append(layer.bias)
        return params
        
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weight = Parameter(np.random.uniform(-limit, limit, (input_dim, output_dim)))
        self.bias = Parameter(np.zeros(output_dim))

    def forward(self, X):
        self.input = X
        return X.dot(self.weight.data) + self.bias.data

    def backward(self, grad_output):
        self.weight.grad = self.input.T.dot(grad_output)
        self.bias.grad = np.sum(grad_output, axis=0)
        return grad_output.dot(self.weight.data.T)

class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

# -------------------- 하이퍼파라미터 및 클래스 초기화 --------------------
learning_rate = 0.1  # 학습률 조정
epochs = 100
input_dim = X_train.shape[1]
hidden_layers = [3, 4]
output_dim = 10
model = CustomMLP(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim)
criterion = BinaryCrossEntropyLoss()
optimizer = SGDOptimizer(params=model.parameters(), learning_rate=learning_rate)

# -------------------- 모델 학습 --------------------
batch_size = 64  # 배치 크기 설정
num_batches = int(np.ceil(X_train.shape[0] / batch_size))

theta_history = []
cost_history = []

for epoch in range(epochs):
    epoch_loss = 0  # 초기화 위치 수정
    # 미니 배치 학습
    for batch_index in range(num_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, X_train.shape[0])
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]
        
        optimizer.zero_grad()
        y_pred = model.forward(X_batch)
        loss = criterion(y_pred, y_batch)
        grad_output = criterion.backward()
        model.backward(grad_output)
        optimizer.step()
        
        epoch_loss += loss

        print(f'Epoch {epoch + 1}, Batch {batch_index + 1}/{num_batches}, Loss: {loss:.4f}', end='\r')
    
    epoch_loss /= num_batches
    cost_history.append(epoch_loss)
    
    # 에포크 로그 출력 (줄바꿈 포함)
    print(f'\nEpoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss:.4f}')
    
    # 에포크당 로스 계산 및 기록
    y_pred_epoch = model.forward(X_train)
    loss_epoch = criterion(y_pred_epoch, y_train)
    cost_history.append(loss_epoch)
    
    # 가중치 기록
    current_theta = model.layers[-2].weight.data.copy()
    theta_history.append(current_theta)

# -------------------- 시각화 --------------------
# 학습 과정의 손실 값 시각화
plt.figure(figsize=(10, 6))
plt.plot(cost_history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

def calculate_accuracy(model, X, y):
    y_pred = model.forward(X)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(y_pred_labels == y_true_labels)
    return accuracy

# 예측 결과 시각화
def plot_predictions_with_accuracy(model, X, y, num_samples=10):
    accuracy = calculate_accuracy(model, X, y)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    indices = np.random.choice(len(X), num_samples, replace=False)
    for i, ax in zip(indices, axes):
        img = X[i].reshape(28, 28)
        true_label = np.argmax(y[i])
        pred_label = np.argmax(model.forward(X[i].reshape(1, -1)))
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}\nPred: {pred_label}')
        ax.axis('off')
    
    # 전체 정확도 시각화
    plt.suptitle(f'Overall Accuracy: {accuracy * 100:.2f}%', fontsize=16)
    plt.show()

plot_predictions_with_accuracy(model, X_test, y_test)


