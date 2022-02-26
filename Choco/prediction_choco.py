import shutil
import os.path

from data_handling import data_labeling
from data_handling import data_consistency
from data_preprocessing import data_preprocessing_predicting
from model_evaluation import ConLearn
from validation_choco import validate_consistency
from XML_handling import xml_parse_categories


def prediction_choco(settings):
    shutil.copyfile(settings["INPUT_XML"], settings["PROGRESS_XML_FILE_PATH"])

    label_Names, features_Data, prediction_names = xml_parse_categories(settings["PROGRESS_XML_FILE_PATH"],
                                                                        settings["IRRELEVANT_FEATURES"],
                                                                        settings["CATEGORIES_FILE_PATH"])

    while label_Names:
        validation_Data, label_Columns, label_Dict, losses, loss_Weights = data_labeling(label_Names,
                                                                                         settings[
                                                                                             "VALIDATION_FILE_PATH"],
                                                                                         settings["BINARY_FEATURES"])
        training_Data, label_Columns, label_Dict, losses, loss_Weights = data_labeling(label_Names,
                                                                                       settings["TRAINING_FILE_PATH"],
                                                                                       settings["BINARY_FEATURES"])
        if not data_consistency(training_Data, validation_Data):
            return 0

        id = ConLearn.model_exists(settings["MODEL_LIBRARY_FILE_PATH"], label_Names)

        if not id:
            print("Model has to be learned first!")
            return 0

        validation_Input = data_preprocessing_predicting(training_Data, validation_Data)

        if not os.path.exists(settings["OUTPUT_XML_FILE_PATH"] + "\conf_0.xml"):
            progress_xml = settings["PROGRESS_XML_FILE_PATH"]
        else:
            progress_xml = settings["OUTPUT_XML_FILE_PATH"] + "\conf_0.xml"

        ConLearn.model_predict_choco(id, validation_Input, label_Names, label_Dict, prediction_names,
                                     validation_Data, progress_xml, settings["OUTPUT_XML_FILE_PATH"])

        label_Names, features_Data, prediction_names = xml_parse_categories(settings["OUTPUT_XML_FILE_PATH"] +
                                                                            "\conf_0.xml",
                                                                            settings["IRRELEVANT_FEATURES"],
                                                                            settings["CATEGORIES_FILE_PATH"])

    validate_consistency(settings["OUTPUT_XML_FILE_PATH"])
    print("Prediction successfully made!")
    return 1


settings_dict = {
    "TRAINING_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_7250_MainFeatures.csv",
    "VALIDATION_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\ValidationData_7250_MainFeatures.csv",
    "MODEL_LIBRARY_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Models\ModelLibrary.csv",
    "PROGRESS_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Progress\Request.xml",
    "OUTPUT_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs",
    "PREDICTION_REST_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Output\Rest.csv",
    "IRRELEVANT_FEATURES": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\Irrelevant Features.txt",
    "BINARY_FEATURES": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\Binary Features.txt",
    "INPUT_XML": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Request.xml",
    "CATEGORIES_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\Categories.csv"
}

prediction_choco(settings_dict)
