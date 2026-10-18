import numpy as np


X = np.array([
    [1200, 2, 10],
    [1500, 3, 5],
    [1800, 3, 8],
    [2200, 4, 3],
    [2500, 4, 1]
], dtype=float)

y = np.array([42, 50, 61, 68, 79], dtype=float)



# Feature scaling


X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X_scaled = (X - X_mean) / X_std


# Parameters


weights = np.zeros(X.shape[1])
bias = 0.0


# Functions


def predict(X, weights, bias):
    return X @ weights + bias


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def compute_gradients(X, y, y_pred):
    n = len(X)

    error = y_pred - y

    dw = (2 / n) * X.T @ error
    db = (2 / n) * np.sum(error)

    return dw, db



# Training

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):

    predictions = predict(X_scaled, weights, bias)

    loss = mse(y, predictions)

    dw, db = compute_gradients(
        X_scaled,
        y,
        predictions
    )

    weights -= learning_rate * dw
    bias -= learning_rate * db

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | "
            f"Loss: {loss:.4f} | "
            f"Weights: {weights} | "
            f"Bias: {bias:.4f}"
        )


print("\nFinal Model")
print("Weights:", weights)
print("Bias:", bias)



# New prediction


new_house = np.array(
    [2000, 3, 4],
    dtype=float
)

new_house_scaled = (
    new_house - X_mean
) / X_std

prediction = predict(
    new_house_scaled,
    weights,
    bias
)

print(
    f"Prediction for {new_house}: "
    f"{prediction:.2f}"
)
