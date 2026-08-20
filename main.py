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

W = np.random.randn(4,1) * 0.01
B = 0.0

score = np.dot(Z,W) + B
prob_score = 1 / (1 + np.exp(-score))

m = Y.shape[0]

epsilon = 1e-15
ps = np.clip(prob_score, epsilon, 1-epsilon)

loss = -(1/m) * np.sum(Y * np.log(ps) + (1-Y)* np.log(1-ps))

Error = ps - Y

dW = 1/m * np.dot(Z.T, Error)
dB = 1/m * np.sum(Error)

learning_rate = 0.1
W = W - learning_rate*dW
B = B - learning_rate*dB

losses =[]
epochs = 1000
for epoch in range(epochs):
    score = np.dot(Z,W) + B
    prob_score = 1 / (1 + np.exp(-score))

    ps = np.clip(prob_score, epsilon, 1-epsilon)
    loss = -(1/m) * np.sum(Y * np.log(ps) + (1-Y)* np.log(1-ps))

    losses.append(loss)

    Error = ps - Y

    dW = 1/m * np.dot(Z.T, Error)
    dB = 1/m * np.sum(Error)
    W = W - learning_rate*dW
    B = B - learning_rate*dB
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

y_pred = (ps>=0.5).astype(int)
accuracy = np.mean(y_pred == Y) * 100
print(f"\nFinal Accuracy: {accuracy:.2f}%")

