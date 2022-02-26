from XML_handling import training_xml_write

import subprocess
import os


def evaluate(epoch_logs, val_x, predictions, prediction_names, label_dict, features_dict, settings):
    pred_dict = {}
    for i in range(len(prediction_names)):
        pred_dict[prediction_names[i]] = predictions[i]

    prediction_accuracy = {}
    for i in range(predictions[0].shape[0] - 1):
        for key, values in pred_dict.items():
            for prediction_name in prediction_names:
                if prediction_name == key:
                    prediction_accuracy[prediction_name] = max(values[i])
                    break

        prediction_values = {}

        for prediction_name in prediction_names:
            if prediction_name in prediction_accuracy:
                for j in range(len(pred_dict[prediction_name][i])):
                    if pred_dict[prediction_name][i][j] == max(pred_dict[prediction_name][i]):
                        # predict value with highest probability
                        prediction_values[prediction_name] = label_dict[prediction_name][j]

        value_count = 0
        feature_values = {}
        for key, values in features_dict.items():
            for j in range(value_count, len(val_x[i])):
                if val_x[i][j] == 1:
                    feature_values[key] = values[j - value_count]
                    value_count += len(values)
                    break

        training_xml_write(prediction_values, feature_values, settings["PROGRESS_XML_FILE_PATH"],
                           settings["EVALUATE_XML_FILE_PATH"] + "\conf_" + str(i) + ".xml")

    result = subprocess.run(["java", "-jar",
                             r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\ConfigurationChecker.jar",
                             r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs"],
                            capture_output=True, text=True, timeout=400)
    if result.returncode == 0:
        with open('complete_output.csv', 'w', newline='') as output:
            output.write(result.stdout)
        for file in os.listdir(settings["EVALUATE_XML_FILE_PATH"]):
            if file.endswith(".xml"):
                os.remove(os.path.join(settings["EVALUATE_XML_FILE_PATH"], file))
    else:
        print('Failure occurred in configurator!')
        for file in os.listdir(settings["EVALUATE_XML_FILE_PATH"]):
            if file.endswith(".xml"):
                os.remove(os.path.join(settings["EVALUATE_XML_FILE_PATH"], file))
        return

    invalid_prediction = {}
    with open('complete_output.csv', 'r') as output:
        output_lines = output.readlines()
        for line in output_lines:
            for label in label_dict:
                if label in line:
                    if label in invalid_prediction:
                        invalid_prediction[label] += 1
                    else:
                        invalid_prediction[label] = 1

    loss = 0
    for key, value in epoch_logs.items():
        if isinstance(epoch_logs[key], float) and key[-4:] == 'loss' and key != 'val_loss' and \
                key[4:-5] in invalid_prediction and key[4:-5] in label_dict:
            epoch_logs[key] = epoch_logs[key] + (0.5 * invalid_prediction[key[4:-5]])
            loss += (0.5 * invalid_prediction[key[4:-5]])

    for key, value in epoch_logs.items():
        if isinstance(epoch_logs[key], float) and key == 'val_loss':
            epoch_logs[key] += loss
            break

    return epoch_logs
