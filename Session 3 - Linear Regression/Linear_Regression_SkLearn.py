import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
R_value = model.score(X,y)

# Print the model intercept, coefficient values and R^2 score
print("Intercept\t=\t{}\nCoeffs\t\t=\t{}".format(model.intercept_, model.coef_))
print("R^2 score\t\t=\t{}".format(R_value))

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