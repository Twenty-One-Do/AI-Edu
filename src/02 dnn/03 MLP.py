import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation

# -------------------- 데이터 불러오기 및 전처리 --------------------

data_path = './data/employment_data.csv'
data = pd.read_csv(data_path)
X = data[['Experience (years)', 'TOEIC Score']].values
y = data['Employment Status'].values.reshape(-1, 1)

# data_path = './data/plant_data.csv'
# data = pd.read_csv(data_path)
# X = data[['Temperature', 'Next Day Temperature']].values
# y = data['Wilt'].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
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

        # 히든 레이어가 없는 경우 처리
        if hidden_layers == [0, 0]:
            self.layers.append(DenseLayer(prev_dim, output_dim))
            self.layers.append(Sigmoid())  # Output layer activation function
        else:
            for hidden_dim in hidden_layers:
                if hidden_dim > 0:  # 0이 아닌 경우에만 레이어 추가
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
epochs = 2000
input_dim = X_train.shape[1]
hidden_layers = [3, 4]  # 히든 레이어의 수와 차원 수를 설정
output_dim = 1
model = CustomMLP(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim)
criterion = BinaryCrossEntropyLoss()
optimizer = SGDOptimizer(params=model.parameters(), learning_rate=learning_rate)

# -------------------- 모델 학습 --------------------
theta_history = []
cost_history = []
probs_history = []

xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
grid = np.c_[xx.ravel(), yy.ravel()]

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    grad_output = criterion.backward()
    model.backward(grad_output)
    optimizer.step()

    # 로스 계산 및 기록
    cost_history.append(loss)

    # 결정경계 계산 및 기록
    probs = model.forward(grid).reshape(xx.shape)
    probs_history.append(probs)

    #가중치 기록
    current_theta = model.layers[-2].weight.data.copy()
    theta_history.append(current_theta)

# -------------------- 시각화 --------------------
theta_1_vals = np.linspace(-100, 100, 100)
theta_2_vals = np.linspace(-100, 100, 100)
theta_1_grid, theta_2_grid = np.meshgrid(theta_1_vals, theta_2_vals)
cost_vals = np.zeros_like(theta_1_grid)

last_layer_weights = model.layers[-2].weight.data

for i in range(theta_1_grid.shape[0]):
    for j in range(theta_1_grid.shape[1]):
        model.layers[-2].weight.data[0, 0] = theta_1_grid[i, j]
        model.layers[-2].weight.data[1, 0] = theta_2_grid[i, j]
        cost_vals[i, j] = criterion(model.forward(X_train), y_train)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.scatter(X_train[y_train.flatten() == 0][:, 0], X_train[y_train.flatten() == 0][:, 1], color='red', label='Class 0 (train)')
ax1.scatter(X_train[y_train.flatten() == 1][:, 0], X_train[y_train.flatten() == 1][:, 1], color='blue', label='Class 1 (train)')
contourf = None

contour = ax2.contourf(theta_1_grid, theta_2_grid, cost_vals, levels=50, cmap='viridis')
point, = ax2.plot([], [], 'ro', label='Current Theta')
path, = ax2.plot([], [], 'r--', label='Descent Path')
text = ax2.text(0, 0, '', ha='center')


def init():
    point.set_data([], [])
    path.set_data([], [])
    text.set_text('')
    return point, path, text

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def update(frame):
    frame = 5*frame
    global contourf
    current_theta = theta_history[frame]
    
    # 결정 경계 시각화
    probs = probs_history[frame]
    
    if contourf:
        for coll in contourf.collections:
            coll.remove()
    contourf = ax1.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['red', 'blue'])
    
    point.set_data(current_theta[0], current_theta[1])
    path.set_data([h[0] for h in theta_history[:frame+1]], [h[1] for h in theta_history[:frame+1]])
    text.set_position((current_theta[0], current_theta[1]))
    text.set_text(f'Epoch: {frame}, Cost: {cost_history[frame]:.4f}')
    
    return point, path, text, contourf

ani = FuncAnimation(fig, update, frames=epochs//5, init_func=init, blit=False, interval=0.001, repeat=False)

ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Logistic Regression Decision Boundary')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Theta 1')
ax2.set_ylabel('Theta 2')
ax2.set_title('Cost Function Contour with Gradient Descent Path')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()