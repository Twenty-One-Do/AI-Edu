import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 데이터 생성
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 - 3 * X + np.random.randn(100, 1)

# 파라미터 초기화
learning_rate = 0.1
epochs = 100
theta_start = np.random.randn(2, 1)  # 초기 theta 값
speed = 100  # 1~10

# 목적 함수
def predict(X, theta):
    return X.dot(theta)

# 손실 함수 (MSE)
def compute_cost(X, y, theta):
    m = len(y)
    cost = (1/(2*m)) * np.sum(np.square(predict(X, theta) - y))
    return cost

# 손실 함수의 도함수
def compute_gradient(X, y, theta):
    m = len(y)
    gradients = (1/m) * X.T.dot(predict(X, theta) - y)
    return gradients

# 데이터에 1 추가 (편향 항을 위해)
X_b = np.c_[np.ones((100, 1)), X]
theta = theta_start
history = [theta.copy()]
cost_history = [compute_cost(X_b, y, theta)]

# 경사 하강법
for _ in range(epochs):
    gradients = compute_gradient(X_b, y, theta)
    theta = theta - learning_rate * gradients
    history.append(theta.copy())
    cost_history.append(compute_cost(X_b, y, theta))

# 손실 함수 등고선 플롯
theta_0_vals = np.linspace(-10, 10, 400)
theta_1_vals = np.linspace(-10, 10, 400)
theta_0_grid, theta_1_grid = np.meshgrid(theta_0_vals, theta_1_vals)
cost_vals = np.zeros_like(theta_0_grid)

for i in range(theta_0_grid.shape[0]):
    for j in range(theta_0_grid.shape[1]):
        t = np.array([[theta_0_grid[i, j]], [theta_1_grid[i, j]]])
        cost_vals[i, j] = compute_cost(X_b, y, t)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# 데이터 플롯
ax1.scatter(X, y)
line, = ax1.plot([], [], color='red', linestyle='--', label='Regression Line')

# 손실 함수 등고선 플롯
contour = ax2.contourf(theta_0_grid, theta_1_grid, cost_vals, levels=50, cmap='viridis')
point, = ax2.plot([], [], 'ro', label='Current Theta')
path, = ax2.plot([], [], 'r--', label='Descent Path')
text = ax2.text(0, 0, '', ha='center')

def init():
    line.set_data([], [])
    point.set_data([], [])
    path.set_data([], [])
    text.set_text('')
    return line, point, path, text

def update(frame):
    theta = history[frame]
    y_pred = predict(X_b, theta)
    line.set_data(X, y_pred)
    
    theta_0 = theta[0, 0]
    theta_1 = theta[1, 0]
    cost = cost_history[frame]
    
    point.set_data(theta_0, theta_1)
    path.set_data([h[0, 0] for h in history[:frame+1]], [h[1, 0] for h in history[:frame+1]])
    text.set_position((theta_0, theta_1))
    text.set_text(f'({theta_0:.2f}, {theta_1:.2f})')
    
    return line, point, path, text

ani = FuncAnimation(fig, update, frames=range(len(history)), init_func=init, blit=True, interval=5000/speed, repeat=False)

ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression with Gradient Descent')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Theta 0')
ax2.set_ylabel('Theta 1')
ax2.set_title('Cost Function Contour with Gradient Descent Path')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
