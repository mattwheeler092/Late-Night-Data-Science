import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



file_name = "Regression_Data_Single_Feature.csv"

data = pd.read_csv(file_name)

output_feature = 'Output'
input_features = ['Feature 1']

input_data = data[input_features].values

X = np.empty((input_data.shape[0], input_data.shape[1] + 1))

for index, row in enumerate(input_data):
    X[index] = np.concatenate(([1], row), axis = 0)

y = data[output_feature].values

X_transpose = np.transpose(X)

X_inverse = np.linalg.inv(np.dot(X_transpose,X))

coeff = np.dot(np.dot(X_inverse,X_transpose),y)


x_test = np.linspace(0,max(y),400)

y_predict = np.empty(400)

for index, val in enumerate(x_test):
    y_predict[index] = coeff[0] + val * coeff[1]

    
plt.plot(x_test, y_predict)
plt.plot(np.transpose(input_data)[0],np.transpose(y),'ro')

plt.show() 

print(coeff)