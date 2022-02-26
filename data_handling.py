import pandas as pd


def data_labeling(label_names, training_file_path, binary_features):
    pandas_Data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    label_Columns = []
    label_Dict = {}
    losses = {}
    loss_Weights = {}
    for name in label_names:
        label_Columns.append(pandas_Data.pop(name))
    for column in label_Columns:
        label_Dict[column.name] = column.unique()
        loss_Weights[column.name] = 1.0
        with open(binary_features, "r") as binary:
            for lines in binary:
                if lines.strip() == column.name:
                    losses[column.name] = "sparse_categorical_crossentropy"
                    break
                else:
                    losses[column.name] = "categorical_crossentropy"
    for key, value in label_Dict.items():
        label_Dict[key] = sorted(value)
    return pandas_Data, label_Columns, label_Dict, losses, loss_Weights


def training_data_labeling(label_names, training_file_path, binary_features):
    pandas_Data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    pandas_Data = pandas_Data.sample(frac=1).reset_index(drop=True)
    label_Columns = []
    label_Dict = {}
    features_Dict = {}
    losses = {}
    loss_Weights = {}
    for name in label_names:
        label_Columns.append(pandas_Data.pop(name))
    for column in label_Columns:
        label_Dict[column.name] = column.unique()
        loss_Weights[column.name] = 1.0
        with open(binary_features, "r") as binary:
            for lines in binary:
                if lines.strip() == column.name:
                    losses[column.name] = "sparse_categorical_crossentropy"
                    break
                else:
                    losses[column.name] = "categorical_crossentropy"
    for key, value in label_Dict.items():
        label_Dict[key] = sorted(value)

    columns = list(pandas_Data)
    for column in columns:
        if column not in label_names:
            features_Dict[column] = pandas_Data[column].unique()
    for key, value in features_Dict.items():
        features_Dict[key] = sorted(value)

    return pandas_Data, label_Columns, label_Dict, features_Dict, losses, loss_Weights


def data_consistency(pandas_data, features_data):
    consistency = True
    for column in list(pandas_data.items()):
        for data in list(features_data.items()):
            if column[0] == data[0]:
                if not set(data[1].values).issubset(set(column[1].values)):
                    consistency = False
                    print('Inconsistent feature: ' + data[0] + ': ' + data[1].values)
                    return consistency
    return consistency


def data_similarity(pandas_data, features_data):
    deviation = {}
    for index, row in features_data.iterrows():
        for i, r in pandas_data.iterrows():
            compare_result = row.compare(r)
            if compare_result.empty:
                deviation[index] = 0
                break
            if index not in deviation:
                deviation[index] = len(compare_result)
            else:
                if len(compare_result) < deviation[index]:
                    deviation[index] = len(compare_result)
    for i in range(max(deviation.values())):
        same_deviation = [k for k, v in deviation.items() if v == i]
        percentage = len(same_deviation) / len(features_data) * 100
        print("Percentage of configurations with " + str(i) +" deviations = " + str(percentage) + "%.")
    return
