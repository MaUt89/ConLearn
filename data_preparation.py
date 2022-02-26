import os
import csv
import xml.etree.ElementTree as et


def training_data_from_xml_get(source_xml_files):
    data_files = []
    for file in os.listdir(source_xml_files):
        data_files.append(file)
    with open(r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_generated.csv", 'a', newline='') \
            as training_data:
        file_amount = len(data_files)
        file_writer = csv.writer(training_data, delimiter=';')
        for i in range(0, file_amount - 1):
            with open(source_xml_files + "\\" + data_files[i]):
                # headings = []
                data = []
                tree = et.parse(source_xml_files + "\\" + data_files[i])
                root = tree.getroot()
                for items in root:
                    for item in items:
                        irrelevant_item = False
                        with open(r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\Irrelevant Features.txt",
                                  "r") as compare_file:
                            for element in compare_file:
                                if item.attrib['key'] == element.strip():
                                    irrelevant_item = True
                                    break
                        if not irrelevant_item:
                            # headings.append(item.attrib['key'])
                            data.append(item.attrib['value'])
                file_writer.writerow(data)

            # print('File %s of %s added.' % (i+1, file_amount))
    print('Training data ready!\n')
    return


# training_data_from_xml_get(r"C:\UserData\z002p84d\MF4ChocoSolver-main\ConfigurationChecker\confs")

"""
def data_prepare():
    result_count = 0
    for file in os.listdir('Learning Data Input/V1_Reporting/Configuration/Processed/'):
        # manipulate data and create one file per CrossModule
        file_path = 'Learning Data Input/V1_Reporting/Configuration/Processed/' + file
        globals()
        with open(file_path, "r+") as configuration_file:
            find_tender_data = []
            for lines in configuration_file:
                find_tender_data = lines.split("|")
                if len(find_tender_data) > 1:
                    if find_tender_data[1] == "CC_ENDCUSTOMERNAME":
                        break
            if len(find_tender_data) == 3 and find_tender_data[2] == '\n':  # only feature name included of tender data
                content = []
                configuration_file.seek(0, 0)
                for lines in configuration_file:
                    content.append(lines)
                content.pop(0)
                content.pop(0)
                string_content = ""
                configuration_file.write(string_content.join(content))
                tender_data_add(file_path)
            else:  # no tender data included
                tender_data_add(file_path)
        with open(file_path, "r") as configuration_file:
            result_count = result_files_create(configuration_file, result_count)
    # Create training-, validation- and testdata
    learning_data_create()
    return


def tender_data_add(configuration_file_path):
    # add relevant tender data to configuration files
    tender_guid = str
    customer_name = str
    customer_country = str
    with open(configuration_file_path, "r") as configuration_file:
        for lines in configuration_file:
            line = lines.strip().split("|")
            if len(line) > 2:
                if line[1] == "GIS_CONFIGURATIONGUID":
                    configuration_guid = line[2]
                    break
    with open("Learning Data Input/V1_Reporting/TenderConfigurationMatch.csv", "r") as compare_file:
        for lines in compare_file:
            line = lines.strip().split(";")
            if line[1] == configuration_guid:
                tender_guid = line[0]
                break
            else:
                continue
    try:
        tender_file_path = 'Learning Data Input/V1_Reporting/Tender/Processed/' + tender_guid + '.txt'
        with open(tender_file_path, "r") as tender_file:
            for lines in tender_file:
                line = lines.strip().split("|")
                if len(line) > 1:
                    if line[1] == "CC_ENDCUSTOMERNAME":
                        customer_name = lines
                    elif line[1] == "CC_ENDCUSTOMERCOUNTRY":
                        customer_country = lines
                    else:
                        continue
        with open(configuration_file_path, "r+") as configuration_file:
            content = configuration_file.read()
            configuration_file.seek(0, 0)
            configuration_file.write(customer_name + customer_country + content)
    except:
        with open(configuration_file_path, "r+") as configuration_file:
            content = configuration_file.read()
            configuration_file.seek(0, 0)
            configuration_file.write('Feature|CC_ENDCUSTOMERNAME|\n' + 'Feature|CC_ENDCUSTOMERCOUNTRY|\n' + content)
    return


def result_files_create(configuration_file, result_count):
    switchgear_data = []
    switchgear_data_input = True
    for lines in configuration_file:
        line = lines.strip().split("|")
        if len(line) == 1:
            if line[0] == "CROSSMODULE":
                result_count += 1
                count = str(result_count)
                switchgear_data_input = False
                print(count)
            continue
        else:
            irrelavant_item = False
            with open("Learning Data Input/V1_Reporting/Irrelevant Features.txt", "r") as compare_file:
                for element in compare_file:
                    if line[1] == element.strip():
                        irrelavant_item = True
                        break
                if not irrelavant_item:
                    if switchgear_data_input:
                        switchgear_data.append(lines)
                    else:
                        result_path = 'Learning Data Input\Result\Result' + count + '.txt'
                        if not os.path.exists('Learning Data Input\Result'):
                            os.mkdir('Learning Data Input\Result')
                        if not os.path.isfile(result_path):
                            with open(result_path, "w") as result_file:
                                for data in switchgear_data:
                                    result_file.write(data)
                                result_file.write(lines)
                        else:
                            if line[0] == 'PARTSET':
                                line[1] = line[1].replace("-", "")
                                if line[1][0:3] == '123' or line[1][0:3] == '427' or line[1][0:3] == '468' or line[1][
                                                                                                        0:3] == '469':
                                    # remove additional partsets
                                    continue
                                elif line[1][0:6] == '479152' or line[1][0:6] == '479338':
                                    # remove additional partsets
                                    continue
                                else:
                                    # use only 11 digit number
                                    line[1] = line[1][:11]
                                    lines = 'PARTSET|' + line[1] + '\n'
                            with open(result_path, "a") as result_file:
                                result_file.write(lines)
    return result_count


def learning_data_create():
    data_files = []
    for file in os.listdir('Learning Data Input/V1_Reporting/Result'):
        data_files.append(file)
    # randomize data
    random.shuffle(data_files)
    with open('Learning Data Input/TrainingData.csv', 'w', newline='') as training_data:
        column_count = 0
        file_writer = csv.writer(training_data, delimiter=';')
        for i in range(0, round(len(data_files) * 0.6)):
            with open('Learning Data Input/Result/' + data_files[i]) as data_file:
                headings = []
                data = []
                for lines in data_file:
                    line = lines.strip().split("|")
                    if i == 0:
                        if line[0] == 'Feature':
                            headings.append(line[1])
                        else:
                            headings.append(line[0])
                    if line[0] == 'Feature':
                        if '.' in line[2]:
                            line[2] = line[2].rstrip('0').rstrip('.')
                        data.append(line[2])
                    else:
                        data.append(line[1])
                if i == 0:
                    file_writer.writerow(headings)
                    file_writer.writerow(data)
                    column_count = len(headings)
                else:
                    if len(data) == column_count:
                        file_writer.writerow(data)
                    else:
                        print('Error in file: ' + data_files[i])
                        file_writer.writerow(data)
    print('Training data ready!\n')
    with open('Learning Data Input/V1_Reporting/ValidationData.csv', 'w', newline='') as validation_data:
        column_count = 0
        file_writer = csv.writer(validation_data, delimiter=';')
        for i in range(round(len(data_files) * 0.6) + 1, round(len(data_files) * 0.8)):
            with open('Learning Data Input/Result/' + data_files[i]) as data_file:
                headings = []
                data = []
                for lines in data_file:
                    line = lines.strip().split("|")
                    if i == 0:
                        if line[0] == 'Feature':
                            headings.append(line[1])
                        else:
                            headings.append(line[0])
                    if line[0] == 'Feature':
                        if '.' in line[2]:
                            line[2] = line[2].rstrip('0').rstrip('.')
                        data.append(line[2])
                    else:
                        data.append(line[1])
                if i == 0:
                    file_writer.writerow(headings)
                    file_writer.writerow(data)
                    column_count = len(headings)
                else:
                    if len(data) == column_count:
                        file_writer.writerow(data)
                    else:
                        print('Error in file: ' + data_files[i])
                        file_writer.writerow(data)
    print('Validation data ready!\n')
    with open('Learning Data Input/V1_Reporting/TestData.csv', 'w', newline='') as test_data:
        column_count = 0
        file_writer = csv.writer(test_data, delimiter=';')
        for i in range(round(len(data_files) * 0.8) + 1, len(data_files)):
            with open('Learning Data Input/Result/' + data_files[i]) as data_file:
                headings = []
                data = []
                for lines in data_file:
                    line = lines.strip().split("|")
                    if i == 0:
                        if line[0] == 'Feature':
                            headings.append(line[1])
                        else:
                            headings.append(line[0])
                    if line[0] == 'Feature':
                        if '.' in line[2]:
                            line[2] = line[2].rstrip('0').rstrip('.')
                        data.append(line[2])
                    else:
                        data.append(line[1])
                if i == 0:
                    file_writer.writerow(headings)
                    file_writer.writerow(data)
                    column_count = len(headings)
                else:
                    if len(data) == column_count:
                        file_writer.writerow(data)
                    else:
                        print('Error in file: ' + data_files[i])
                        file_writer.writerow(data)
    print('Test data ready!\n')
    return
"""
