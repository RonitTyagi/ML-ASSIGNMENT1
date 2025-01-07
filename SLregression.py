import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading datasets
X = pd.read_csv("linearX.csv", header=None).values
Y = pd.read_csv("linearY.csv", header=None).values

# Normalize the predictor variable X
X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

# Add the intercept term to X
X_b = np.c_[np.ones((len(X_normalized), 1)), X_normalized]

# Cost function calculation
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X.dot(theta)
    errors = predictions - Y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Batch Gradient Descent
def gradient_descent(X, Y, theta, learning_rate, iterations):
    m = len(Y)
    cost_history = []
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - Y
        theta -= (learning_rate / m) * X.T.dot(errors)
        cost_history.append(compute_cost(X, Y, theta))
    return theta, cost_history

# Stochastic Gradient Descent
def stochastic_gradient_descent(X, Y, theta, learning_rate, iterations):
    m = len(Y)
    cost_history = []
    for i in range(iterations):
        for j in range(m):
            rand_index = np.random.randint(0, m)
            X_i = X[rand_index:rand_index+1]
            Y_i = Y[rand_index:rand_index+1]
            prediction = X_i.dot(theta)
            error = prediction - Y_i
            theta -= learning_rate * X_i.T.dot(error)
        cost_history.append(compute_cost(X, Y, theta))
    return theta, cost_history

# Mini-Batch Gradient Descent
def mini_batch_gradient_descent(X, Y, theta, learning_rate, iterations, batch_size):
    m = len(Y)
    cost_history = []
    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]
            predictions = X_batch.dot(theta)
            errors = predictions - Y_batch
            theta -= (learning_rate / batch_size) * X_batch.T.dot(errors)
        cost_history.append(compute_cost(X, Y, theta))
    return theta, cost_history

# Initializing the parameters
learning_rate = 0.5
iterations = 50
batch_size = 32

theta = np.zeros((2, 1))

# Batch Gradient Descent
theta_bgd, cost_history_bgd = gradient_descent(X_b, Y, theta.copy(), learning_rate, iterations)

# Stochastic Gradient Descent
theta_sgd, cost_history_sgd = stochastic_gradient_descent(X_b, Y, theta.copy(), learning_rate, iterations)

# Mini-Batch Gradient Descent
theta_mbgd, cost_history_mbgd = mini_batch_gradient_descent(X_b, Y, theta.copy(), learning_rate, iterations, batch_size)

# Plot cost function vs iterations
plt.figure(figsize=(12, 6))
plt.plot(range(iterations), cost_history_bgd, label="Batch Gradient Descent")
plt.plot(range(iterations), cost_history_sgd, label="Stochastic Gradient Descent")
plt.plot(range(iterations), cost_history_mbgd, label="Mini-Batch Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function vs Iterations for Gradient Descent Methods")
plt.legend()
plt.show()

# Plot dataset and fitted line using Batch Gradient Descent parameters
def plot_regression_line(X, Y, theta, label):
    plt.scatter(X, Y, color="blue", label="Data Points")
    plt.plot(X, theta[0] + theta[1] * (X - X_mean) / X_std, color="red", label=label)
    plt.xlabel("Predictor Variable")
    plt.ylabel("Response Variable")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

plot_regression_line(X, Y, theta_bgd, "Regression Line (BGD)")

# Test with different learning rates
learning_rates = [0.005, 0.5, 5]

plt.figure(figsize=(14, 8))  # Larger figure size for clarity
for lr in learning_rates:
    theta = np.zeros((2, 1))
    _, cost_history = gradient_descent(X_b, Y, theta, lr, iterations)
    plt.plot(range(iterations), cost_history, label=f"Learning Rate: {lr}", marker='o', markersize=4, linewidth=2)

# Enhance graph features
plt.yscale('log')  # Use logarithmic scale for better visibility
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Cost (Log Scale)", fontsize=14)
plt.title("Cost Function vs Iterations for Different Learning Rates", fontsize=16, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()  # Adjusted the layout for better readability
plt.show()

def plot_regression_line(X, Y, theta, label):
    plt.scatter(X, Y, color="blue", label="Data Points")  # Plot data points
    plt.plot(X, theta[0] + theta[1] * (X - X_mean) / X_std, color="red", label=label)  # Plot regression line
    plt.xlabel("Predictor Variable")
    plt.ylabel("Response Variable")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

# Plot the regression line from Batch Gradient Descent (using theta_bgd obtained from gradient descent)
plot_regression_line(X, Y, theta_bgd, "Regression Line (BGD)")

# Final Costs
final_cost_bgd = compute_cost(X_b, Y, theta_bgd)
final_cost_sgd = compute_cost(X_b, Y, theta_sgd)
final_cost_mbgd = compute_cost(X_b, Y, theta_mbgd)

print("Final Costs:")
print(f"Batch Gradient Descent: {final_cost_bgd}")
print(f"Stochastic Gradient Descent: {final_cost_sgd}")
print(f"Mini-Batch Gradient Descent: {final_cost_mbgd}")
