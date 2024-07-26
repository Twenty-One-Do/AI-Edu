import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation

# 데이터 불러오기
data_path = './data/student_admission_data.csv'
data = pd.read_csv(data_path)

# 입력값과 결과값 분리
X = data[['IQ', '공부 시간 (시간)']].values
y = data['합격 여부'].values

# 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 손실 함수 정의 (로지스틱 회귀의 비용 함수)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5
    cost = -(1/m) * (y.T.dot(np.log(h + epsilon)) + (1 - y).T.dot(np.log(1 - h + epsilon)))
    return cost

# 손실 함수의 도함수 (경사)
def compute_gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    gradient = (1/m) * X.T.dot(h - y)
    return gradient

# 학습 데이터에 편향 항 추가
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# 파라미터 초기화
learning_rate = 2.0
epochs = 1000
theta = np.random.randn(X_train_bias.shape[1])
history = [theta.copy()]
cost_history = [compute_cost(X_train_bias, y_train, theta)]

# 경사 하강법
for _ in range(epochs):
    gradient = compute_gradient(X_train_bias, y_train, theta)
    theta -= learning_rate * gradient
    history.append(theta.copy())
    cost_history.append(compute_cost(X_train_bias, y_train, theta))

# 손실 함수 등고선 플롯
theta_1_vals = np.linspace(-1, 10, 100)
theta_2_vals = np.linspace(-1, 10, 100)
theta_1_grid, theta_2_grid = np.meshgrid(theta_1_vals, theta_2_vals)
cost_vals = np.zeros_like(theta_1_grid)

for i in range(theta_1_grid.shape[0]):
    for j in range(theta_1_grid.shape[1]):
        t = np.array([theta[0], theta_1_grid[i, j], theta_2_grid[i, j]])
        cost_vals[i, j] = compute_cost(X_train_bias, y_train, t)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# 데이터 플롯
ax1.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0 (train)')
ax1.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 1 (train)')
contourf = None

# 손실 함수 등고선 플롯
contour = ax2.contourf(theta_1_grid, theta_2_grid, cost_vals, levels=50, cmap='viridis')
point, = ax2.plot([], [], 'ro', label='Current Theta')
path, = ax2.plot([], [], 'r--', label='Descent Path')
text = ax2.text(0, 0, '', ha='center')

def init():
    point.set_data([], [])
    path.set_data([], [])
    text.set_text('')
    return point, path, text

def update(frame):
    global contourf
    current_theta = history[frame]
    
    # 결정 경계 시각화
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_bias = np.c_[np.ones((grid.shape[0], 1)), grid]
    probs = sigmoid(grid_bias.dot(current_theta)).reshape(xx.shape)
    
    if contourf:
        for coll in contourf.collections:
            coll.remove()
    contourf = ax1.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['red', 'blue'])
    
    point.set_data(current_theta[1], current_theta[2])
    path.set_data([h[1] for h in history[:frame+1]], [h[2] for h in history[:frame+1]])
    text.set_position((current_theta[1], current_theta[2]))
    text.set_text(f'Epoch: {frame}, Cost: {cost_history[frame]:.4f}')
    
    return point, path, text, contourf

ani = FuncAnimation(fig, update, frames=range(len(cost_history)), init_func=init, blit=False, interval=50, repeat=False)

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
