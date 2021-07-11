import shutil

from data_handling import data_labeling
from data_handling import data_consistency
from data_preprocessing import data_preprocessing_predicting
from XML_handling import xml_parse
from XML_handling import xml_write
from previous_prediction import check_previous_prediction
from previous_prediction import write_next_neighbor_value
from nearest_neighbor import predict_classification


def prediction(settings):
    shutil.copyfile(settings["INPUT_XML"], settings["PROGRESS_XML_FILE_PATH"])
    label_Names, features_Data = xml_parse(settings["PROGRESS_XML_FILE_PATH"], settings["IRRELEVANT_FEATURES"])

    if not label_Names:
        return 0

    previous_prediction = check_previous_prediction(label_Names, settings["PREDICTION_REST_FILE_PATH"])
    if previous_prediction:
        prediction_values = write_next_neighbor_value(previous_prediction, settings["PREDICTION_REST_FILE_PATH"],
                                                      settings["TRAINING_FILE_PATH"])
        if prediction_values:
            xml_write(prediction_values, settings["PROGRESS_XML_FILE_PATH"], settings["OUTPUT_XML_FILE_PATH"])
            return 1
        else:
            return 0

    pandas_Data, label_Columns, label_Dict, losses, loss_Weights = data_labeling(label_Names,
                                                                                 settings["TRAINING_FILE_PATH"],
                                                                                 settings["BINARY_FEATURES"])
    if not data_consistency(pandas_Data, features_Data):
        return 0

    test_Input = data_preprocessing_predicting(pandas_Data, features_Data)

    prediction_values = predict_classification(test_Input, test_Input[test_Input.shape[0] - 1], label_Names,
                                               settings["TRAINING_FILE_PATH"], settings["PREDICTION_REST_FILE_PATH"])

    xml_write(prediction_values, settings["PROGRESS_XML_FILE_PATH"], settings["OUTPUT_XML_FILE_PATH"])

    print("Prediction successfully made!")
    return 1


"""
settings_dict = {
    "TRAINING_FILE_PATH": "Learning Data Input/V2_XML/TrainingData_MainFeatures.csv",
    "MODEL_LIBRARY_FILE_PATH": "Models/ModelLibrary.csv",
    "PROGRESS_XML_FILE_PATH": "Learning Data Input/V2_XML/XML Input/Progress/Request.xml",
    "OUTPUT_XML_FILE_PATH": "Learning Data Output/Answer.xml",
    "PREDICTION_REST_FILE_PATH": "Learning Data Output/Rest.csv",
    "IRRELEVANT_FEATURES": "Learning Data Input/V2_XML/Irrelevant Features_MainFeatures.txt",
    "BINARY_FEATURES": "Learning Data Input/V2_XML/Binary Features.txt",
    "INPUT_XML": "Learning Data Input/V2_XML/XML Input/Request.xml",
    "CATEGORIES_FILE_PATH": "Learning Data Input/V2_XML/Categories.csv"
}

prediction(settings_dict)
"""
