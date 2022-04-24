# Example of making predictions
import os
import csv
import pandas as pd
import subprocess

from XML_handling import prediction_xml_write
from XML_handling import configuration_xml_write
from XML_handling import solver_xml_parse
from math import sqrt



class NearestNeighbor:
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(row1, row2, variable_values):
        distance = 0.0
        for var, val in row1.iteritems():
            for variable, value in row2.iteritems():
                if var == variable:
                    if val == value:
                        break
                    else:
                        distance += variable_values[var]
        return sqrt(distance)

    # Locate the most similar neighbors
    def get_neighbors(training_data, test_row, variable_values):
        distances = list()
        i = 0
        for index, train_row in training_data.iterrows():
            dist = NearestNeighbor.euclidean_distance(test_row, train_row, variable_values)
            distances.append((i, dist))
            i += 1
            if dist == 0:  # remove if nn > 1
                break

        distances.sort(key=lambda tup: tup[1])
        neighbors = []

        for i in range(0, 1):
            neighbors.append(distances[i][0])

        return neighbors

    # Make a classification prediction with neighbors
    def determine_nearest_neighbor(training_data, test_row, label_names, variable_values):
        neighbor_position = NearestNeighbor.get_neighbors(training_data, test_row, variable_values)
        prediction_values = {}
        neighbor_values = {}
        for neighbor in neighbor_position:
            neighbor_data = training_data.iloc[neighbor]
            for i in range(len(label_names)):
                if not neighbor_values or not label_names[i] in neighbor_values:
                    neighbor_values[label_names[i]] = [neighbor_data[label_names[i]]]
                else:
                    neighbor_values[label_names[i]].append(neighbor_data[label_names[i]])
        for i in range(len(neighbor_values)):
            prediction_values[label_names[i]] = max(set(neighbor_values[label_names[i]]),
                                                    key=neighbor_values[label_names[i]].count)

        """
        # save sorted other neighbors to file to apply if prediction is invalid
        with open(rest_file_path, "w", newline='') as rest_file:
            writer = csv.writer(rest_file, delimiter=';')
            name = [label_names[0], 'Distance']
            writer.writerow(name)
            for pos, dis in rest:
                item = [pos, dis]
                writer.writerow(item)
        """
        return prediction_values

    def predict_nearest_neighbor(training_file_path, validation_data, prediction_names, progress_xml, output_file_path):
        training_data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
        variable_values = {}
        for index, item in training_data.iteritems():
            variable_values[item.name] = item.nunique()
        for i in range(len(validation_data)):
            prediction = NearestNeighbor.determine_nearest_neighbor(training_data, validation_data.iloc[i],
                                                                    prediction_names, variable_values)
            print(str(i) + ": " + str(prediction))
            prediction_xml_write(prediction, validation_data, i, progress_xml,
                                 output_file_path + "\conf_" + str(i) + ".xml")
        return

    # Make a classification prediction with neighbors for solver
    def determine_nearest_neighbor_solver(training_data, test_row, label_names, variable_values):
        neighbor_position = NearestNeighbor.get_neighbors(training_data, test_row, variable_values)
        prediction_values = {}
        neighbor_values = {}
        for neighbor in neighbor_position:
            neighbor_data = training_data.iloc[neighbor]
            for i in range(len(label_names)):
                if not neighbor_values or not label_names[i] in neighbor_values:
                    neighbor_values[label_names[i]] = [neighbor_data[label_names[i]]]
                else:
                    neighbor_values[label_names[i]].append(neighbor_data[label_names[i]])

        prediction_order = {}
        for i in range(len(neighbor_values)):
            for item in neighbor_values[label_names[i]]:
                # if item == max(set(neighbor_values[label_names[i]]),
                                                          # key=neighbor_values[label_names[i]].count):
                    # prediction_values[label_names[i]] = item
                if not prediction_order or not label_names[i] in prediction_order:
                    prediction_order[label_names[i]] = [[item], [1]]
                else:
                    if not item in prediction_order[label_names[i]][0]:
                        prediction_order[label_names[i]][0].append(item)
                        prediction_order[label_names[i]][1].append(1)
                    else:
                        for j in range(len(prediction_order[label_names[i]][0])):
                            if prediction_order[label_names[i]][0][j] == item:
                                prediction_order[label_names[i]][1][j] += 1

        for prediction_name in prediction_order:
            zipped_lists = zip(prediction_order[prediction_name][1], prediction_order[prediction_name][0])
            sorted_pairs = sorted(zipped_lists, reverse=True)
            tuples = zip(*sorted_pairs)
            prediction_order[prediction_name] = [list(tuple) for tuple in tuples]
            prediction_values[prediction_name] = prediction_order[prediction_name][1][0]

        return prediction_values, prediction_order

    def predict_nn_solver(training_file_path, validation_data, prediction_names, progress_xml,
                          output_file_path):
        training_data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
        variable_values = {}
        for index, item in training_data.iteritems():
            variable_values[item.name] = item.nunique()
        for i in range(len(validation_data)):
            prediction, prediction_order = NearestNeighbor.determine_nearest_neighbor_solver(training_data, validation_data.iloc[i],
                                                                           prediction_names,
                                                                           variable_values)

            with open(output_file_path + "\Solver\VariableProbability.csv", 'w', newline='') as file:
                file_writer = csv.writer(file, delimiter=';')
                for variable in prediction_order:
                    line = [variable]
                    for j in range(len(prediction_order[variable][0])):
                        line.append(prediction_order[variable][1][j])
                    file_writer.writerow(line)

            """with open(output_file_path + "\Solver\GivenVariables.csv", 'w') as file:
                file_writer = csv.writer(file, delimiter=';')
                file_writer.writerow(validation_data.loc[[i]])"""

            configuration_xml_write(validation_data.iloc[i], progress_xml,
                                    output_file_path + "\Solver\conf_withoutPrediction.xml")

            try:
                result = subprocess.run(["java", "-jar",
                                         r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\conf_identifier.jar",
                                         r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs\Solver\VariableProbability.csv",
                                         r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs\Solver\conf_withoutPrediction.xml",
                                         "1"],
                                        capture_output=True, text=True, timeout=100)

                if result.returncode == 0:
                    with open('solver_output.csv', 'w', newline='') as output:
                        output.write(result.stdout)
                else:
                    print('Failure occurred in configurator!')
            except:
                print('Subprocess did not answer! Continue with another try...')

            prediction_values = solver_xml_parse(r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Data\conf_1.xml",
                                                 prediction_names)
            print(str(i) + ": " + str(prediction_values))
            prediction_xml_write(prediction_values, validation_data, i, progress_xml,
                                 output_file_path + "\conf_" + str(i) + ".xml")

            for file in os.listdir(output_file_path + "\Solver"):
                os.remove(os.path.join(output_file_path + "\Solver", file))
        return



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
prediction = predict_nearest_neighbor(dataset, dataset[0], 3)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
"""
