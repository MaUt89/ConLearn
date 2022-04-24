import subprocess
import os
import pandas as pd
import re

from XML_handling import xml_parse


def validate_consistency(validation_file_path):
    validation_count = len([f for f in os.listdir(validation_file_path) if
                            os.path.isfile(os.path.join(validation_file_path, f))])
    inconsistent_predictions = 0
    consistency = 0
    try:
        result = subprocess.run(["java", "-jar",
                                r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\checker.jar",
                                r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs"],
                                capture_output=True, text=True, timeout=400)

        if result.returncode == 0:
            with open('complete_output.csv', 'w', newline='') as output:
                output.write(result.stdout)
        else:
            print('Failure occurred in configurator!')
            for file in os.listdir(validation_file_path):
                if file.endswith(".xml"):
                    os.remove(os.path.join(validation_file_path, file))
            return


        with open('output', 'r') as output:
            output_lines = output.readlines()
            for line in output_lines:
                if 'inconsistent' in line:
                    inconsistent_predictions += 1
                    # os.remove(os.path.join(validation_file_path, line.split(";")[0]))
        consistency = round(((validation_count - inconsistent_predictions) / validation_count) * 100, 2)
        print("Average consistency of prediction is: " + str(consistency) + "%")
    except:
        print('Subprocess did not answer! Continue with another try...')
    """
    for file in os.listdir(validation_file_path):
        if file.endswith(".xml"):
            os.remove(os.path.join(validation_file_path, file))
    """
    return consistency


def validate_accuracy(prediction_file_path, validation_file_path, irrelevant_features, prediction_names):
    count = 0
    accuracy = 0
    overall_wrong_predictions = {}
    for file in os.listdir(prediction_file_path):
        if file.endswith(".xml"):
            file_path = prediction_file_path + "\\" + file
            not_predicted_variables, prediction = xml_parse(file_path, irrelevant_features)
            validation_data = pd.read_csv(validation_file_path, delimiter=';', dtype='string')
            temp = re.findall(r'\d+', file)
            file_number = list(map(int, temp))
            correct_prediction = 0
            wrong_predictions = {}

            for column in prediction:
                for col in validation_data:
                    if column in prediction_names and column == col and prediction.iloc[0][column] == validation_data.iloc[file_number[0]][col]:
                        correct_prediction += 1
                        break
                    else:
                        if column in prediction_names and column == col:
                            wrong_predictions[column] = prediction.iloc[0][column], validation_data.iloc[file_number[0]][col]

            if not overall_wrong_predictions:
                for key, value in wrong_predictions.items():
                    overall_wrong_predictions[key + ": " + value[0] + ", " + value[1]] = 1
            else:
                for key, value in wrong_predictions.items():
                    if key + ": " + value[0] + ", " + value[1] in overall_wrong_predictions.keys():
                        overall_wrong_predictions[key + ": " + value[0] + ", " + value[1]] = overall_wrong_predictions[key + ": " + value[0] + ", " + value[1]] + 1
                    else:
                        overall_wrong_predictions[key + ": " + value[0] + ", " + value[1]] = 1

            accuracy = accuracy + (correct_prediction / len(prediction_names) * 100)
            count += 1

    overall_wrong_predictions = dict(sorted(overall_wrong_predictions.items(), key=lambda item: item[1]))
    for key, value in overall_wrong_predictions.items():
        print(key + ": " + str(value))
    overall_accuracy = accuracy / count
    print("Average accuracy of prediction is: " + str(overall_accuracy) + "%")
    return overall_accuracy
