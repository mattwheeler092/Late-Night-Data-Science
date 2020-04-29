import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self, fit_intercept = True):
        """Initialisation function for the LinearRegression class.
        Class Attributes:
            self.fit_intercept {bool} -- Boolean to indicate if the regression model should fit an intercept or not (default: {True})
            self.coeff_ {ndarray} -- numpy array to hold the linear coefficient values (default: empty array)
            self.intercept_ {float} -- Holds the fitted intercept value (default: 0)
        """
        self.fit_intercept = fit_intercept
        self.coeff_ = np.array([])
        self.intercept_ = 0.0
        
    def fit(self, X, y): 
        """Function to fit the linear regession model to the input data
        Arguments:
            X {ndarray (N x M)} -- 2d numpy array holding the M feature values for the N input data points
            y {ndarray (N x 1)} -- 1d numpy array holding the output values
        """
        if self.fit_intercept:
            D = np.empty((X.shape[0], X.shape[1] + 1))
            for index, row in enumerate(X):
                D[index] = np.concatenate(([1], row), axis = 0)
            D_inverse = np.linalg.inv(np.dot(D.T,D))
            coeff = np.dot(np.dot(D_inverse,D.T),y)
            self.coeff_ = coeff[1:]
            self.intercept_ = coeff[0]
        else:
            D = X
            D_inverse = np.linalg.inv(np.dot(D.T,D))
            self.coeff_ = np.dot(np.dot(D_inverse,D.T),y)
    
    def predict(self, X):
        """Function to predict the outcome value from the input feature data
        Arguments:
            X {ndarray (N x M)} -- 2d numpy array holding the M feature values for each of the N input data points
        Raises:
            ValueError: Error raised when regression model hasn't been fit to the data
        Returns:
            ndarray (1 x N) -- Returns the predicted output values for the N input data points
        """
        if self.coeff_.size == 0:
            raise ValueError("No data has been fit to the linear regression model.")
        prediction = np.ones(X.shape[0]) * self.intercept_
        for index, row in enumerate(X):
            prediction[index] += np.dot(self.coeff_, row)
        return prediction
    
    
    def score(self, X, y, adjusted = True):
        """Function to compute the R^2 score the fitted regression model
        Arguments:
            X {ndarray (N x M)} -- 2d numpy array holding the M feature values for the N input data points
            y {ndarray (N x 1)} -- 1d numpy array holding the true output values to compare against
            adjusted {bool} -- Boolean to indicate if the score is the adjusted R^2 score or just R^2 score (default: {True})
        Raises:
            ValueError: Error raised when the regression model hasn't been fit to any data
        Returns:
            float -- Returns the R^2 score for the input data
        """
        if self.coeff_.size == 0:
            raise ValueError("No data has been fit to the linear regression model.")
        y_pred = self.predict(X)
        u = sum(list(map(lambda val : val ** 2, y - y_pred)))
        v = sum(list(map(lambda val : val ** 2, y - np.mean(y))))
        n = X.shape[0]
        k = self.coeff_.size
        if adjusted:
            return 1 - (((n - 1)/(n - k - 1)) * (1 - u / v))
        else:
            return 1 - u / v



def main():

    # Load in the data to be fitted
    file_name = "Regression_Data_Single_Feature.csv"
    data = pd.read_csv(file_name)

    # Indicate which columns correspond to the output feature (y) and the predictor features (X)
    y_feature = 'Output'
    X_features = ['Feature 1']

    # Pull output and predictor data from dataframe
    y = data[y_feature].values
    X = data[X_features].values
    
    # Create a linear regression model object
    model = LinearRegression(fit_intercept = True)
    
    # Fit the linear regression model to the loaded data
    model.fit(X,y)
    
    # Compute the R^2 score using the loaded data
    R_value = model.score(X, y, adjusted = True)
    
    # Print the model intercept, coefficient values and R^2 score
    print("Intercept\t=\t{}\nCoeffs\t\t=\t{}".format(model.intercept_, model.coeff_))
    print("R^2 score\t=\t{}".format(R_value))
    
    """ CODE FOR PLOTTING THE RESULTS FOR 2D DATA """
    
    # Create a linearly distributed sample of predictor values
    X_pred = np.linspace(0,max(y),100)
    X_pred = np.reshape(X_pred,(100,1))
    
    # Predict the output values of the linear distribution
    y_pred = model.predict(X_pred)
    
    # Plot the fitted data along with the regression line
    plt.plot(X_pred, y_pred, label = "Linear Regression Model")
    plt.plot(X.T[0], y.T, 'o', label = "Original Data")
    plt.ylabel("Outcome Feature")
    plt.xlabel("Predictor Feature")
    plt.title("Linear Regression Model Plot")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
