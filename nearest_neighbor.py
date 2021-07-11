# Example of making predictions
import csv
import pandas as pd
from math import sqrt


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row):
    distances = list()
    i = 0
    for train_row in train:
        if i == train.shape[0]-2:   # last entry needs to be ignored since it is the row to be predicted
            continue
        dist = euclidean_distance(test_row, train_row)
        distances.append((i, dist))
        i += 1

    distances.sort(key=lambda tup: tup[1])
    neighbors = []

    for i in range(0, 20):
        neighbors.append(distances[i][0])
        distances.pop(i)

    return neighbors, distances


# Make a classification prediction with neighbors
def predict_classification(train, test_row, label_names, training_file_path, rest_file_path):
    neighbor_position, rest = get_neighbors(train, test_row)
    training_Data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    prediction_values = {}
    neighbor_values = []
    for neighbor in neighbor_position:
        neighbor_data = training_Data.iloc[neighbor]
        neighbor_values.append(neighbor_data[label_names[0]])

    prediction_values[label_names[0]] = max(set(neighbor_values), key=neighbor_values.count)
    print(prediction_values)

    # save sorted other neighbors to file to apply if prediction is invalid
    with open(rest_file_path, "w", newline='') as rest_file:
        writer = csv.writer(rest_file, delimiter=';')
        name = [label_names[0], 'Distance']
        writer.writerow(name)
        for pos, dis in rest:
            item = [pos, dis]
            writer.writerow(item)

    return prediction_values

"""
# Test distance function
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
prediction = predict_classification(dataset, dataset[0], 3)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
"""