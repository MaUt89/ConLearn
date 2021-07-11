import os
import csv
import pandas as pd


def check_previous_prediction(label_names, prediction_rest_file_path):
    try:
        with open(prediction_rest_file_path, "r") as prediction_rest:
            for line in prediction_rest:
                data = line.split(";")
                if data[0] in label_names:
                    # last predicted label is saved in first column
                    return data[0]
        os.remove(prediction_rest_file_path)
        return
    except IOError:
        print("There has not been a previous prediction. Continuing with new prediction!")
        return


def write_next_possible_value(previous_prediction, prediction_rest_file_path):
    prediction_values = {}
    with open(prediction_rest_file_path, "r") as prediction_rest:
        for line in prediction_rest:
            data = line.split(";")
            if len(data) > 1:
                # second column has first information about prediction
                prediction = data[-1].rstrip('\n')
                break
            else:
                return

    df = pd.read_csv(prediction_rest_file_path, delimiter=';', dtype='string')
    # remove from file because value is predicted in this loop
    df.drop(prediction, axis=1, inplace=True)
    df.to_csv(prediction_rest_file_path, index=False, sep=';')

    prediction_values[previous_prediction] = prediction
    print(prediction_values)

    return prediction_values


def write_next_neighbor_value(previous_prediction, rest_file_path, training_file_path):
    file = []
    data_position = ''
    with open(rest_file_path, "r") as prediction_rest:
        for line in prediction_rest:
            data = line.split(";")
            if data[0] == previous_prediction or data[0] == '':
                data[1] = data[1].rstrip('\n')
                file.append(data)
                continue
            else:
                if not data_position:
                    data_position = data[0]
                    continue
                data[1] = data[1].rstrip('\n')
                file.append(data)

    with open(rest_file_path, "w", newline='') as rest_file:
        writer = csv.writer(rest_file, delimiter=';')
        for item in file:
            writer.writerow(item)

    training_Data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    neighbor = training_Data.iloc[int(data_position)]
    prediction_values = {previous_prediction: neighbor[previous_prediction]}
    print(prediction_values)
    return
