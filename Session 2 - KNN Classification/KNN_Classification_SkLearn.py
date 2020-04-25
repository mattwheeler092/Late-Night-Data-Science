from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load in the data
data = pd.read_csv("Classification_Dataset.csv")

# Select data column to act at the label and predictor fields
Label = "Label"                              
Predictors = ["Predictor_1","Predictor_2","Predictor_3","Predictor_4"]    

# Select program parameters: 
k = 5                      # Number of nearest neighbours to consider
train_fraction = 0.8       # Fraction of data to include in training dataset

# Split the data randomly into test and train data
data["Split_Value"] = np.random.rand(data[Label].values.size)
train_data = data[data["Split_Value"] < train_fraction].copy()
test_data = data[data["Split_Value"] > train_fraction].copy()

# Create a list of unique labels
label_list = sorted(list(set(data[Label].values)))

# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors = k)

# Train the classifier using the training dataset
knn.fit(train_data[Predictors].values, train_data[Label].values)

# Create a list of the actual labels of the testing dataset
actual_label = test_data[Label].values

# Predict the labels for the testing dataset
predicted_label = knn.predict(test_data[Predictors].values)

# Produce a confusion matrix from the prediction
confusion_matrix = confusion_matrix(actual_label, predicted_label, labels = label_list)

# Create a dataframe from the confusion matrix
df_cm = pd.DataFrame(confusion_matrix, index = label_list, columns = label_list)

# Plot the confusion matrix
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot = True, cmap = "Blues")
plt.xlabel("Predicted Label\n", fontsize = 14)
plt.ylabel("Actual Label\n", fontsize = 14)
plt.title("Confusion Matrix of KNN Classifier\n", fontsize = 16)
plt.show()

# Compute and print the precision, recall and F1 score values for each label
precision, recall, f1_score, count = precision_recall_fscore_support(actual_label, predicted_label)
for index, label in enumerate(label_list):
    print("Classification Label: {}".format(label))
    print("\tPrecision\t=\t{:.2f}".format(precision[index]))
    print("\tRecall\t\t=\t{:.2f}".format(recall[index]))
    print("\tF1 Score\t=\t{:.2f}".format(f1_score[index]))

