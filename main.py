import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(25)

samples = 200
study_hours = np.random.uniform(1,10,200)
attendence = np.random.uniform(40,100,200)
past_mark = np.random.uniform(40,100,200)
assign_marks = np.random.uniform(40,100,200)

tot_score = (0.3 * (study_hours*10)) + (0.3 * attendence) + (0.2 * past_mark) + (0.2 * assign_marks)

res = (tot_score>65).astype(int)

data = pd.DataFrame({"Study Hours": study_hours , "Attendence": attendence , "Past Mark": past_mark , "Assignment Marks": assign_marks , "Result": res})

X = data[["Study Hours", "Attendence", "Past Mark", "Assignment Marks"]].values
Y = data[["Result"]].values.reshape(-1,1)


X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

Z = (X-X_mean)/X_std

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

W1 = np.random.randn(4, 8) * 0.01
B1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1) * 0.01
B2 = np.zeros((1, 1))

m = Y.shape[0]
epsilon = 1e-15
learning_rate = 0.1
losses =[]
epochs = 1000


for epoch in range(epochs):
    Z1 = np.dot(Z, W1) + B1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + B2
    A2 = sigmoid(Z2)

    
    ps = np.clip(A2, epsilon, 1 - epsilon)
    loss = -(1 / m) * np.sum(Y * np.log(ps) + (1 - Y) * np.log(1 - ps))
    losses.append(loss)

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    dB2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(Z.T, dZ1)
    dB1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

y_pred = (A2 >= 0.5).astype(int)
accuracy = np.mean(y_pred == Y) * 100
print(f"\nFinal Accuracy: {accuracy:.2f}%")

new_student = np.array([[8, 85, 75, 80]])
new_student_scaled = (new_student - X_mean) / X_std

z1_new = np.dot(new_student_scaled, W1) + B1
a1_new = relu(z1_new)

z2_new = np.dot(a1_new, W2) + B2
a2_new = sigmoid(z2_new)

prob = a2_new[0][0]
prediction = "PASS" if prob >= 0.5 else "FAIL"
print(f"New Student Prediction: {prediction} (Probability: {prob * 100:.2f}%)")