import shutil

from data_handling import training_data_labeling
from data_handling import data_consistency
from data_preprocessing import data_preprocessing_learning
from model_evaluation import ConLearn
from XML_handling import xml_parse_categories
from clean_up import clean_up


def learn(settings):
    shutil.copyfile(settings["INPUT_XML"], settings["PROGRESS_XML_FILE_PATH"])

    label_Names, features_Data, prediction_names = xml_parse_categories(settings["PROGRESS_XML_FILE_PATH"],
                                                                        settings["IRRELEVANT_FEATURES"],
                                                                        settings["CATEGORIES_FILE_PATH"])
    if not label_Names:
        return 0

    pandas_Data, label_Columns, label_Dict, features_Dict, losses, loss_Weights = training_data_labeling(label_Names,
                                                                                 settings["TRAINING_FILE_PATH"],
                                                                                 settings["BINARY_FEATURES"])
    if not data_consistency(pandas_Data, features_Data):
        return 0

    train_X, test_X, train_Labels, test_Labels = data_preprocessing_learning(pandas_Data, label_Columns)
    model = ConLearn.build_model(train_X.shape[1], label_Dict)
    id = ConLearn.model_evaluation(model, losses, loss_Weights, train_X, test_X, train_Labels, test_Labels,
                                   label_Dict, features_Dict, prediction_names, settings)
    ConLearn.save_model_csv(id, settings["TRAINING_FILE_PATH"], label_Names, settings["MODEL_LIBRARY_FILE_PATH"])

    return 1


settings_dict = {
    "TRAINING_FILE_PATH": "Learning Data Input/V2_XML/TrainingData_MainFeatures.csv",
    "MODEL_LIBRARY_FILE_PATH": "Models/ModelLibrary.csv",
    "PROGRESS_XML_FILE_PATH": "Learning Data Input/V2_XML/XML Input/Progress/Request.xml",
    "ANSWER_XML_FILE_PATH": "C:\Temp\Answer.xml",
    "EVALUATE_XML_FILE_PATH": "C:\Temp\Evaluate.xml",
    "IRRELEVANT_FEATURES": "Learning Data Input/V2_XML/Irrelevant Features_MainFeatures.txt",
    "BINARY_FEATURES": "Learning Data Input/V2_XML/Binary Features.txt",
    "INPUT_XML": "Learning Data Input/V2_XML/XML Input/Request.xml",
    "CATEGORIES_FILE_PATH": "Learning Data Input/V2_XML/Categories.csv"
}

learn(settings_dict)
clean_up(settings_dict)
