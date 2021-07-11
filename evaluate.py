from XML_handling import training_xml_write
from XML_handling import xml_parse_compare

import subprocess


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
                           settings["EVALUATE_XML_FILE_PATH"])
        result = subprocess.run([r"C:\Program Files\camos\camosWinClient.exe", "-knb=gis_Configurations",
                                 "-ver=w", "-DBSettings=gis_DBSettings_QM", "-NoPKILogin",
                                 "-host=https://sn1t4801.ad101.siemens-energy.net/camosApps",
                                 "-start=gis_TestWebService", "-report", "-trace",
                                 "-path=[Client]C:\Temp\Evaluate.xml"],
                                capture_output=True, text=True)

        if result.returncode == 0:
            invalid_prediction = xml_parse_compare(settings["ANSWER_XML_FILE_PATH"], settings["EVALUATE_XML_FILE_PATH"],
                                                   settings["IRRELEVANT_FEATURES"])
            loss = 0
            for key, value in epoch_logs.items():
                if isinstance(epoch_logs[key], float) and key[-4:] == 'loss' and key != 'val_loss' and \
                        key[4:-5] in invalid_prediction and key[4:-5] in prediction_names:
                    epoch_logs[key] = epoch_logs[key] + 0.1
                    loss += epoch_logs[key]

            for key, value in epoch_logs.items():
                if isinstance(epoch_logs[key], float) and key == 'val_loss':
                    epoch_logs[key] = loss
                    break
        else:
            print('Failure occurred in configurator!')

    return epoch_logs
