import os


from flask import Flask, request, send_from_directory
# from prediction_neighbor import prediction
from prediction import prediction
from clean_up import clean_up


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


flask_app = Flask(__name__)


@flask_app.route("/ConLearn", methods=['GET'])
def welcome():
    return "Welcome to ConLearn!"


@flask_app.route("/ConLearn/Predict", methods=['GET'])
def predict():
    if prediction(settings_dict):
        return "Prediction successfully made!", 200
    else:
        return "Bad request!", 400


@flask_app.route("/ConLearn/XML/Answer", methods=['GET'])
def XMLAnswer():
    return send_from_directory(settings_dict['OUTPUT_XML_FILE_PATH'].replace('/Answer.xml', ''), 'Answer.xml',
                               as_attachment=True)


@flask_app.route("/ConLearn/XML/Request", methods=['POST'])
def XMLRequest():
    with open(os.path.join(settings_dict['INPUT_XML'].replace('/Request.xml', ''), 'Request.xml'), "wb") as fp:
        fp.write(request.data)
        if os.path.isfile(settings_dict['INPUT_XML']):
            return "", 201
        else:
            return "Bad request!", 400


@flask_app.route("/ConLearn/Cleanup", methods=['GET'])
def clean():
    if clean_up(settings_dict):
        return "Clean up ready!", 200
    else:
        return "Bad request!", 400


if __name__ == "__main__":
    flask_app.run(debug=True)
