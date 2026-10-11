import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([42, 50, 61, 68, 79], dtype=float)

w= 0.0
b=0.0

def predict(X, w,b):
    return w*X + b

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

predictions = predict(X,w,b)
loss = mse(y,predictions)
print(loss)

def gradient_computation(X,y,y_pred):
    n= len(X)
    error = y_pred - y

    dw = (2/n) * np.sum(error*X) 
    db = (2/n) * np.sum(error)

    return dw, db



learning_rate = 0.01



times =1000

for time in range(times):
    predictions = predict(X, w,b)

    loss=mse(y,predictions)

    dw, db= gradient_computation(X,y,predictions)

    w -= learning_rate*dw
    b -= learning_rate*db

    if(time%100==0):
        print(
            f"Times : {time} |"
            f"Loss : {loss} |"
            f"w = {w:.4f} |"
            f"b = {b:.4f} "
        )

print("\nFinal Model:")
print(f"y = {w:0.4f}x + {b:0.4f}")

new_x= 6
prediction =predict(new_x,w,b)
print(f"Prediction for x ={new_x} : {prediction:.2f}")
