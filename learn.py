import shutil
import os.path
import data_handling
import timeit


from data_preprocessing import data_preprocessing_learning
from model_evaluation import ConLearn
from nearest_neighbor import NearestNeighbor
from XML_handling import xml_parse_categories
from data_preprocessing import data_preprocessing_predicting
from Choco.validation_choco import validate_consistency
from Choco.validation_choco import validate_accuracy
from clean_up import clean_up


def learn(settings):
    models_to_learn = 1

    model_performance ={}
    model_accuracy = {}
    model_consistency = {}

    shutil.copyfile(settings["INPUT_XML"], settings["PROGRESS_XML_FILE_PATH"])

    label_Names, features_Data, prediction_names = xml_parse_categories(settings["PROGRESS_XML_FILE_PATH"],
                                                                        settings["IRRELEVANT_FEATURES"],
                                                                        settings["CATEGORIES_FILE_PATH"])
    if not label_Names:
        return 0

    validation_Data, label_Columns, label_Dict, losses, loss_Weights = data_handling.data_labeling(label_Names,
                                                                                     settings["VALIDATION_FILE_PATH"],
                                                                                     settings["BINARY_FEATURES"])

    # learn the models
    for i in range(models_to_learn):

        pandas_Data, label_Columns, label_Dict, features_Dict, losses, loss_Weights = data_handling.training_data_labeling(label_Names,
                                                                                                                           prediction_names,
                                                                                                                           settings["TRAINING_FILE_PATH"],
                                                                                                                           settings["BINARY_FEATURES"])
        if not data_handling.data_consistency(pandas_Data, features_Data) \
                or not data_handling.data_consistency(pandas_Data, validation_Data):
            return 0

        # calculate similarity of training dataset and validation dataset
        # data_handling.data_similarity(pandas_Data, validation_Data)
        train_X, test_X, train_Labels, test_Labels, input_neuron_list, output_neuron_list = data_preprocessing_learning(pandas_Data,
                                                                                                                        label_Columns)
        model = ConLearn.build_model(train_X.shape[1], label_Dict, input_neuron_list, output_neuron_list,
                                     settings["KNOWLEDGE_BASE_RULES"])
        id = ConLearn.model_evaluation(model, losses, loss_Weights, train_X, test_X, train_Labels, test_Labels,
                                   label_Dict, features_Dict, prediction_names, settings)
        ConLearn.save_model_csv(id, settings["TRAINING_FILE_PATH"], label_Names, settings["MODEL_LIBRARY_FILE_PATH"])

    # evaluate the quality of the learned models
    validation_Input = data_preprocessing_predicting(pandas_Data, validation_Data)

    if not os.path.exists(settings["OUTPUT_XML_FILE_PATH"] + "\conf_0.xml"):
        progress_xml = settings["PROGRESS_XML_FILE_PATH"]
    else:
        progress_xml = settings["OUTPUT_XML_FILE_PATH"] + "\conf_0.xml"

    for i in range(models_to_learn):
        id = ConLearn.model_id_get(settings["MODEL_LIBRARY_FILE_PATH"], i + 0)
        starttime = timeit.default_timer()
        #ConLearn.model_predict_choco(id, validation_Input, label_Names, label_Dict, prediction_names,
                                     #validation_Data, progress_xml, settings["OUTPUT_XML_FILE_PATH"])
        ConLearn.model_predict_solver(id, validation_Input, label_Names, label_Dict, prediction_names,
                                      validation_Data, progress_xml, settings["OUTPUT_XML_FILE_PATH"],
                                      settings["FEATURE_COMPLEXITY_ORDER"])
        # NearestNeighbor.predict_nearest_neighbor(settings["TRAINING_FILE_PATH"], validation_Data, prediction_names, progress_xml,
                                 # settings["OUTPUT_XML_FILE_PATH"])
        # NearestNeighbor.predict_nn_solver(settings["TRAINING_FILE_PATH"], validation_Data, prediction_names, progress_xml,
                                 # settings["OUTPUT_XML_FILE_PATH"])
        stoptime = timeit.default_timer()
        average_runtime = (stoptime - starttime) / len(validation_Data)
        model_accuracy[id] = validate_accuracy(settings["OUTPUT_XML_FILE_PATH"], settings["VALIDATION_FILE_PATH"],
                                               settings["IRRELEVANT_FEATURES"], prediction_names)
        model_consistency[id] = validate_consistency(settings["OUTPUT_XML_FILE_PATH"])
        model_performance[id] = (model_accuracy[id] + model_consistency[id]) / 2

    # remove all models except the best
    id = ConLearn.model_cleanup(settings["MODEL_LIBRARY_FILE_PATH"], model_performance)

    for key in model_consistency.keys():
        if key != id:
            shutil.rmtree('Models/' + key)

    return print("Selected model achieved " + str(model_performance[id]) + "% performance and has been added to model library! \n The accuracy of the model was "
                 + str(model_accuracy[id]) + "%, whereas the consistency reached a value of " + str(model_consistency[id]) + "%! \n"
                 + "Average runtime was: " + str(average_runtime))


settings_dict = {
    "TRAINING_FILE_PATH": "Learning Data Input/V2_XML/TrainingData_100000_RuleFeatures.csv",
    "VALIDATION_FILE_PATH": "Learning Data Input/V2_XML/ValidationData_725_RuleFeatures.csv",
    "MODEL_LIBRARY_FILE_PATH": "Models/ModelLibrary.csv",
    "PROGRESS_XML_FILE_PATH": "Learning Data Input/V2_XML/XML Input/Progress/Request.xml",
    "OUTPUT_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs",
    "EVALUATE_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs",
    "IRRELEVANT_FEATURES": "Learning Data Input/V2_XML/Irrelevant Features_RuleFeatures.txt",
    "BINARY_FEATURES": "Learning Data Input/V2_XML/Binary Features.txt",
    "INPUT_XML": "Learning Data Input/V2_XML/XML Input/Request_0.xml",
    "CATEGORIES_FILE_PATH": "Learning Data Input/V2_XML/Categories_RuleFeatures.csv",
    "FEATURE_COMPLEXITY_ORDER": "Learning Data Input/V2_XML/Feature_Complexity_Order.csv",
    "KNOWLEDGE_BASE_RULES": "rules_v6.txt"
}

learn(settings_dict)
# clean_up(settings_dict)
