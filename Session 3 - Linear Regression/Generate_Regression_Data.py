from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(435545)

num_featues = 3
num_samples = 250
variance = 50.0

# Generate fetures, outputs, and true coefficient of 100 samples,
features, output = make_regression(n_samples = num_samples,
                                         # three features
                                         n_features = num_featues,
                                         # where only two features are useful,
                                         n_informative = num_featues,
                                         # 0.0 standard deviation of the guassian noise
                                         noise = variance)

feature_labels = []

features = np.reshape(features, (num_featues, len(features)))

for index, array in enumerate(features):
    array -= min(array)
    array *= max(output) / max(array)
    
    feature_labels.append("Feature {}".format(index + 1))

features = np.reshape(features, (len(features[0]),num_featues))

rand_num = np.random.randint(0,100)

output -= min(output) - rand_num

data = pd.DataFrame(features, columns = feature_labels)

data['Output'] = output

print(data.head())

data.to_csv('Regression_Data_Multiple_Features.csv', index = False)


