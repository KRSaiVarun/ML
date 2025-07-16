import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# 1. Create sample data (1000 points with some noise)
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)

# 2. Make and train the model
model = LinearRegression().fit(X, y)

# 3. Make predictions
predictions = model.predict(X)

# 4. Show results
print("Model score:", model.score(X, y))

# 5. Plot the data and line
plt.scatter(X, y)
plt.plot(X, predictions, 'r-')
plt.show()
