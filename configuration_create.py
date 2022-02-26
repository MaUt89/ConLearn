import pandas as pd
import shutil
import csv
import os

from datetime import datetime
from random import choices
from XML_handling import configuration_xml_write
from Choco.validation_choco import validate_consistency
from data_preparation import training_data_from_xml_get


def configuration_create(settings_dict):
    configurations_added = 0
    for i in range(100):
        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Configuration creation started!\n")

        num_configurations = 1000
        shutil.copyfile(settings_dict["INPUT_XML"], settings_dict["PROGRESS_XML_FILE_PATH"])
        pandas_Data = pd.read_csv(settings_dict["CONFIGURATION_FILE_PATH"], delimiter=';', dtype='string')
        original_configurations_num = pandas_Data.shape[0]

        value_distribution = {}
        for column in pandas_Data:
            var_values = pandas_Data[column].unique()
            var_distribution = {}
            for var in var_values:
                var_distribution[var] = 0
            configuration_count = 0
            for item in pandas_Data[column]:
                for var in var_values:
                    if var == item:
                        var_distribution[var] += 1
                configuration_count += 1
            for item in var_distribution:
                var_distribution[item] = var_distribution[item] / configuration_count
            value_distribution[column] = var_distribution

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Value distribution determined! Next: Determine randomized values!\n")

        original_configuration_data = pandas_Data.head(723)

        for j in range(num_configurations + original_configurations_num):
            if j >= original_configurations_num:
                configuration = original_configuration_data.sample()

                for item in value_distribution:
                    change_value = choices([0, 1], [0.95, 0.05])
                    if change_value[0]:
                        configuration[item] = choices(list(value_distribution[item].keys()),
                                                      list(value_distribution[item].values()))[0]

                pandas_Data = pandas_Data.append(configuration, ignore_index=True)

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Randomized values determined! Next: Create configurations!\n")

        for index, row in pandas_Data.iterrows():
            if index >= original_configurations_num:
                configuration_xml_write(row, settings_dict["PROGRESS_XML_FILE_PATH"],
                                        settings_dict["OUTPUT_XML_FILE_PATH"] + "\conf_" + str(index) + ".xml")

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Configurations created! Next: Validate configurations!\n")

        consistency = validate_consistency(settings_dict["OUTPUT_XML_FILE_PATH"])
        if consistency == 0:
            continue

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Configurations validated! Next: Extend training file!\n")

        valid_configurations = 0
        with open('output', 'r', newline='') as output_file:
            configurations_checked = csv.reader(output_file, delimiter=';')
            for row in configurations_checked:
                if row[1] == 'inconsistent':
                    if os.path.isfile(settings_dict["OUTPUT_XML_FILE_PATH"] + "\\" + row[0]):
                        os.remove(settings_dict["OUTPUT_XML_FILE_PATH"] + "\\" + row[0])
                else:
                    valid_configurations += 1

        training_data_from_xml_get(settings_dict["OUTPUT_XML_FILE_PATH"])

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Configurations successfully created and validated! "
              "Training file has been extended by " + str(valid_configurations) + " valid configurations")
        configurations_added += valid_configurations
    return print("Successfully finished and " + str(configurations_added)+" configurations added!")


settings_dict = {
    "CONFIGURATION_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_new.csv",
    "INPUT_XML": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Request.xml",
    "PROGRESS_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Progress\Request.xml",
    "OUTPUT_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs"
}

configuration_create(settings_dict)
