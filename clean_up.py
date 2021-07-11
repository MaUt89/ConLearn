import os


def clean_up(settings):
    try:
        os.remove(settings["INPUT_XML"])
        print("Input file has been removed!")
    except:
        print("Input file has already been removed!")

    try:
        os.remove(settings["OUTPUT_XML_FILE_PATH"])
        print("Output file has been removed!")
    except:
        print("Output file has already been removed!")

    try:
        os.remove(settings["PROGRESS_XML_FILE_PATH"])
        print("Progress file has been removed!")
    except:
        print("Progress file has already been removed!")

    try:
        os.remove(settings["PREDICTION_REST_FILE_PATH"])
        print("Rest file has been removed!")
    except:
        print("Rest file has already been removed!")

    try:
        os.remove(settings["TRAINING_XML_FILE_PATH"])
        print("Training file has been removed!")
    except:
        print("Training file has already been removed!")

    try:
        os.remove(settings["EVALUATE_XML_FILE_PATH"])
        print("Evaluation file has been removed!")
    except:
        print("Evaluation file has already been removed!")

    try:
        os.remove(settings["ANSWER_XML_FILE_PATH"])
        print("Answer file has been removed!")
    except:
        print("Answer file has already been removed!")
    return 1
