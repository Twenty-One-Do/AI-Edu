import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#↓↓↓↓↓↓↓↓↓↓↓↓↓값을 바꿔보세요!↓↓↓↓↓↓↓↓↓↓↓↓↓

# 함수와 파라미터를 바꿔보면서 local minimum으로 가는지 확인해보세요
def f(x):
    return 0.1*x**4-2*x**2+x+15

def df(x):
    return 0.4*x**3-4*x+1

learning_rate = 0.1
epochs = 100
x_start = 5
x_definition = (-5,5)
speed = 5 # 1~10
#↑↑↑↑↑↑↑↑↑↑↑↑값을 바꿔보세요!↑↑↑↑↑↑↑↑↑↑↑↑

x = x_start
history = [x]


#↓↓↓↓↓↓↓↓↓↓↓↓↓경사하강법!↓↓↓↓↓↓↓↓↓↓↓↓↓
for _ in range(epochs):
    gradient = df(x)
    x = x - learning_rate * gradient
#↑↑↑↑↑↑↑↑↑↑↑↑AI 학습의 핵심!↑↑↑↑↑↑↑↑↑↑↑↑
    history.append(x)

fig, ax = plt.subplots(figsize=(10, 6))
x_vals = np.linspace(*x_definition, 400)
y_vals = f(x_vals)

ax.plot(x_vals, y_vals)
scat = ax.scatter([], [], color='red')
line, = ax.plot([], [], color='red', linestyle='--', label='Gradient Descent Path')
text = ax.text(0, 0, '', ha='center')
tangent_line, = ax.plot([], [], color='blue', linestyle='-', label='Tangent Line')
gradient_text_red = ax.text(0, 0, '', ha='center', color='red', visible=False)
gradient_text_blue = ax.text(0, 0, '', ha='center', color='blue', visible=False)

def init():
    scat.set_offsets(np.empty((0, 2)))
    line.set_data([], [])
    text.set_text('')
    tangent_line.set_data([], [])
    gradient_text_red.set_text('')
    gradient_text_blue.set_text('')
    return scat, line, text, tangent_line, gradient_text_red, gradient_text_blue

def update(frame):
    x = history[frame]
    y = f(x)
    gradient = df(x)
    
    tangent_x = np.linspace(x - 1, x + 1, 100)
    tangent_y = gradient * (tangent_x - x) + y
    
    scat.set_offsets(np.array([[x, y]]))
    line.set_data(history[:frame+1], [f(x) for x in history[:frame+1]])
    text.set_position((x, y))
    text.set_text(f'({x:.2f}, {y:.2f})')
    
    tangent_line.set_data(tangent_x, tangent_y)
    if gradient >= 0:
        gradient_text_red.set_text(f'Gradient: {gradient:.2f}')
        gradient_text_red.set_position((x, y + 1))
        gradient_text_red.set_visible(True)
        gradient_text_blue.set_visible(False)
    else:
        gradient_text_blue.set_text(f'Gradient: {gradient:.2f}')
        gradient_text_blue.set_position((x, y + 1))
        gradient_text_blue.set_visible(True)
        gradient_text_red.set_visible(False)
    
    return scat, line, text, tangent_line, gradient_text_red, gradient_text_blue

ani = FuncAnimation(fig, update, frames=range(len(history)), init_func=init, blit=True, interval=5000/speed, repeat=False)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Visualization with Tangent Lines and Gradients')
plt.legend()
plt.grid(True)
plt.show()
