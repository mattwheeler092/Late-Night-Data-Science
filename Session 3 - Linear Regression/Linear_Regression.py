import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.coef_ = []
        self.intercept_ = 0.0
        
    def fit(self, X, y): 
        if self.fit_intercept:
            D = np.empty((X.shape[0], X.shape[1] + 1))
            for index, row in enumerate(X):
                D[index] = np.concatenate(([1], row), axis = 0)
            D_inverse = np.linalg.inv(np.dot(D.T,D))
            coeff = np.dot(np.dot(D_inverse,D.T),y)
            self.coef_ = coeff[1:]
            self.intercept_ = coeff[0]
        else:
            D = np.copy(X)
            D_inverse = np.linalg.inv(np.dot(D.T,D))
            self.coef_ = np.dot(np.dot(D_inverse,D.T),y)
    
    def predict(self, X):
        if not self.coef_:
            raise ValueError("No data has been fit to the linear regression model.")
        prediction = np.ones(X.shape[0]) * self.intercept_
        for index, row in enumerate(X):
            prediction[index] += np.dot(self.coef_, row)
        return prediction
    
    
    def score(self, X, y):
        if not self.coef_:
            raise ValueError("No data has been fit to the linear regression model.")
        y_pred = self.predict(X)
        u = sum(list(map(lambda val : val ** 2, y - y_pred)))
        v = sum(list(map(lambda val : val ** 2, y - np.mean(y))))
        return (1 - u / v)



def main():

    file_name = "Regression_Data_Single_Feature.csv"

    data = pd.read_csv(file_name)

    y_feature = 'Output'
    X_features = ['Feature 1']

    y = data[y_feature].values
    X = data[X_features].values
    
    model = LinearRegression()
    model.fit(X,y)
    
    X_pred = np.linspace(0,max(y),100)
    y_pred = model.predict(X_pred)
    
    plt.plot(X_pred, y_pred)
    plt.plot(X.T[0], y.T, 'o')
    plt.show()
    
    
if __name__ == "__main__":
    main()

"""
X = np.empty((input_data.shape[0], input_data.shape[1] + 1))

for index, row in enumerate(input_data):
    X[index] = np.concatenate(([1], row), axis = 0)

y = data[output_feature].values

X_inverse = np.linalg.inv(np.dot(X.T,X))

coeff = np.dot(np.dot(X_inverse,X.T),y)


x_test = np.linspace(0,max(y),400)

y_predict = np.empty(400)

for index, val in enumerate(x_test):
    y_predict[index] = coeff[0] + val * coeff[1]

    
plt.plot(x_test, y_predict)
plt.plot(input_data.T[0],y.T,'ro')

plt.show() 

print(coeff)
"""