import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 데이터 불러오기
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv(url, names=column_names, delim_whitespace=True, na_values='?')

# 결측치 처리
data = data.dropna()

# 특성과 타겟 변수 설정
X = data[['horsepower']].values  # 예시로 horsepower 사용
y = data['mpg'].values

# 학습 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 다항 특성 생성
poly = PolynomialFeatures(degree=2)  # 2차 다항식
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 다항 회귀 모델 학습
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 예측
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# 모델 성능 평가
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(poly.transform(X_range))
plt.plot(X_range, y_range_pred, color='red', label='Polynomial Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Polynomial Regression for Auto MPG Dataset')
plt.legend()
plt.show()

# 사용자 입력을 받아 MPG 예측
horsepower_input = float(input("Enter the horsepower value: "))
horsepower_poly = poly.transform(np.array([[horsepower_input]]))
mpg_prediction = model.predict(horsepower_poly)
print(f"The predicted MPG for a car with {horsepower_input} horsepower is {mpg_prediction[0]:.2f}")
