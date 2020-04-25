import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""------------------------------------------------------------------------------------------"""

def compute_distance_array(test_row_data, train_data):
    """ DEFINITION: 
            Function to determine the distances between a test data point and all of the train data points
        PARAMETERS:
            test_row_data   -   (1 x L numpy array) Holds the L predictor values for the test data point
            train_data      -   (M x L numpy array) Holds the L predictor values for the M training 
                                data rows
        OUTPUTS:
            distance_array  -   (M x 1 list) Holds distance values between test point and the M 
                                training data points
    """
    distance_array = []
    for train_row_data in train_data:
        diff = train_row_data - test_row_data
        sqr_diff = list(map(lambda x: x ** 2, diff))
        distance_array.append(pow(sum(sqr_diff),0.5))
    return distance_array

"""------------------------------------------------------------------------------------------"""

def KNN_validation(data, Label, Predictors, k_neighbours = 1, train_fraction = 0.8):
    """ DEFINITION: 
            Function to test how effectively a dataset can be used to perform KNN classification.
            Splits the data into test and trian, classifies the test data and produces a resulting
            confusion matrix.
        PARAMETERS:
            data                -   (pandas dataframe) Dataframe to contain the data
            Label               -   (string) Name of column to be used as the category labels
            Predictors          -   (list) List of column names to be used as predictor values
            k_neighbours        -   (int) Number of nearest neightbours to consider [Defaulted to 1]
            train_fraction      -   (float) Fraction of dataframe used as training data [Defaulted to 0.8]
        OUTPUTS:
            confusion_matrix    -   (N x N numpy array) Confusion matrix for the N unique labels
            label_list          -   (1 x N list) Contains list of the N unique labels as they apper in 
                                    the confusion_matrix
    """
    # Split the data randomly into test and train data
    data["Split_Value"] = np.random.rand(data[Label].values.size)
    train_data = data[data["Split_Value"] < train_fraction].copy()
    test_data = data[data["Split_Value"] > train_fraction].copy()

    # Create a list of unique labels
    label_list = sorted(list(set(data[Label].values)))

    # Create an empty confusion matrix
    confusion_matrix = np.zeros([len(label_list),len(label_list)])

    # For each test data point in the test dataframe:
    for _, row in test_data.iterrows():

        # Create column in the train dataframe containing the distances to the test data point
        train_data["Distances"] = compute_distance_array(row[Predictors].values,train_data[Predictors].values)

        # Sort the dataframe to show the closest points at the top
        train_data = train_data.sort_values("Distances")

        # Create a list of the labels of the 'k_neighbours' number of nearest neighbours
        neighbour_labels = train_data[Label].values[:k_neighbours]

        # Create a list of unique labels that appear in the above list
        unique_labels = list(set(neighbour_labels))

        # If all neightbours share the same label: 
        if len(unique_labels) == 1:

            # Set the test data points predicted label to be the same 
            predicted_label = train_data[Label].values[0]

            # Increase the relevant confusion_matrix element by 1
            confusion_matrix[label_list.index(row[Label])][label_list.index(predicted_label)] += 1

        # Else, if the neighbours don't share a common label:
        else:
            # Count how may times each label occurs in the 'neighboutr_labels' list
            count = list(map(lambda x : np.count_nonzero(neighbour_labels == x), unique_labels))

            # Combine the labels with their occurance counts
            label_count = dict(zip(unique_labels, count))

            # Create a list of labels that appear the most times in the nearest neightbours
            mode_labels = list(filter(lambda x : label_count[x] >= max(count), label_count))

            # If there is only one most common label:
            if len(mode_labels) == 1:

                # Set the test data points predicted label to be the same 
                predicted_label = mode_labels[0]

                # Increase the relevant confusion_matrix element by 1
                confusion_matrix[label_list.index(row[Label])][label_list.index(predicted_label)] += 1

            # If more than one label appears the most number of times: 
            else:
                # Randly set the test data points predicted label to be one of these labels
                predicted_label = np.random.choice(mode_labels)

                # Increase the relevant confusion_matrix element by 1
                confusion_matrix[label_list.index(row[Label])][label_list.index(predicted_label)] += 1

    # Return the populated confusion matrix
    return confusion_matrix, label_list

"""------------------------------------------------------------------------------------------"""

def print_confusion_matrix(confusion_matrix, label_list, normalise = False):
    """ DEFINITION:
            Function to print a confusion matrix in a graphical format. Printed using either explicit
            numbers or normalised view of numbers.
        PARAMETERS:
            confusion_matrix    -   (N x N numpy array) Populated confusion matrix
            label_list          -   (1 x N list) List of N categories as they apper in the confusion_matrix
            normalise           -   (Boolean) If 'True' the confusion_matrix will be normalised along rows
        OUTPUTS:
            No function outputs
    """
    if normalise:
        for index, row in enumerate(confusion_matrix):
            confusion_matrix[index] = row / sum(row)
    dataframe = pd.DataFrame(confusion_matrix, index = label_list, columns = label_list)
    plt.figure(figsize = (10,7))
    sns.heatmap(dataframe, annot = True, cmap = "Blues")
    plt.xlabel("Predicted Label\n", fontsize = 14)
    plt.ylabel("Actual Label\n", fontsize = 14)
    plt.title("Confusion Matrix of KNN Classifier\n", fontsize = 16)
    plt.show()
    return 0

"""------------------------------------------------------------------------------------------"""

def analyse_confusion_matrix(confusion_matrix, label_list):
    """ DEFINITION:
            Function to analyse a confusion matrix and print the precision, recall and F1 values
            for each of the categories
        PARAMETERS:
            confusion_matrix    -   (N x N numpy array) Populated confusion matrix
            label_list          -   (1 x N list) List of N categories as they apper in the confusion_matrix
        OUTPUTS:
            No function outputs
    """
    for index in range(len(label_list)):
        true_positive = 0
        false_negative = 0
        false_positive = 0
        true_negative = 0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix)):
                if i == index and j == index:
                    true_positive += confusion_matrix[i][j]
                elif i == index and j != index:
                    false_negative += confusion_matrix[i][j]
                elif i != index and j == index:
                    false_positive += confusion_matrix[i][j]
                else:
                    true_negative += confusion_matrix[i][j]
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = (2 * precision * recall) / (precision + recall)
        print("Classification Label: {}".format(label_list[index]))
        print("\tPrecision\t=\t{:.2f}".format(precision))
        print("\tRecall\t\t=\t{:.2f}".format(recall))
        print("\tF1 Score\t=\t{:.2f}".format(f1_score))
    return 0

"""------------------------------------------------------------------------------------------"""
"""-----------------------------------  MAIN FUNCTION  --------------------------------------"""

def main():
    
    # Load in the data
    data = pd.read_csv("Classification_Dataset.csv")

    # Select data column to act at the label and predictor fields
    Label = "Label"                              
    Predictors = ["Predictor_1","Predictor_2","Predictor_3","Predictor_4"]

    # Select program parameters: 
    k = 5                   # Number of nearest neighbours to consider
    train_split = 0.8       # Fraction of data to include in training dataset

    # Perform validation on the loaded dataset to produce resulting confusion matrix
    confusion_matrix, label_list  = KNN_validation(data, Label, Predictors, k, train_split)

    # Analyse the confusion matrix to determine corresponding precision, recall and F1 score values
    analyse_confusion_matrix(confusion_matrix, label_list)

    # Plot the confusion matrix
    print_confusion_matrix(confusion_matrix, label_list, False)

    return 0

if __name__ == "__main__":
    main()



