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