import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Car Price Prediction.csv')

# Features and target
features = ["horsepower", "enginesize", "curbweight", "citympg"]
X = df[features].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
print("Linear Regression -> RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lin)),
      " R2:", r2_score(y_test, y_pred_lin))

# Polynomial Regression (degree 2, 3, 4)
for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    r2 = r2_score(y_test, y_pred_poly)
    print(f"Polynomial (deg={degree}) -> RMSE: {rmse}, R2: {r2}")

# Plot fitted curves (single feature: horsepower vs price)
X_single = df[["horsepower"]].values
y_single = df["price"].values
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y_single, test_size=0.2, random_state=42)
x_range = np.linspace(X_single.min(), X_single.max(), 200).reshape(-1, 1)

plt.scatter(X_single, y_single, color="blue", alpha=0.5, label="Actual Data")

lin_s = LinearRegression().fit(X_train_s, y_train_s)
plt.plot(x_range, lin_s.predict(x_range), label="Linear (deg=1)")

for degree, color in zip([2, 3, 4], ["red", "green", "purple"]):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly_s = poly.fit_transform(X_train_s)
    model = LinearRegression().fit(X_train_poly_s, y_train_s)
    y_curve = model.predict(poly.transform(x_range))
    plt.plot(x_range, y_curve, label=f"Polynomial (deg={degree})", color=color)

plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Linear vs Polynomial Regression Fits")
plt.legend()
plt.show()