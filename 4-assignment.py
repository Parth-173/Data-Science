import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X = housing.data[:, 0].reshape(-1, 1)
y = housing.target 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='yellow', label='Actual Data')

sorted_indices = np.argsort(X_test.flatten())
plt.plot(X_test.flatten()[sorted_indices],
         y_pred[sorted_indices],
         color='green', linewidth=2, label='Regression Line')

# Label the axes and add a title
plt.xlabel(housing.feature_names[0])
plt.ylabel('Median House Value')
plt.title('Linear Regression on California Housing Data')
plt.legend()
plt.show()
