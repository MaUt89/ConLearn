from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def data_preprocessing_learning(pandas_data, label_columns):
    data_array = np.array(pandas_data)
    one_hot = OneHotEncoder()
    input_neuron_list = {}
    fit_array = one_hot.fit(data_array)
    count = 0
    for category in fit_array.categories_:
        n_list = []
        for item in category:
            n_list.append(item)
        input_neuron_list[pandas_data.columns[count]] = n_list
        count += 1

    data_one_hot = one_hot.fit_transform(data_array).toarray()
    print(data_one_hot)
    label_one_hot = []
    output_neuron_list = {}
    label_binarizer = LabelBinarizer()

    for column in label_columns:
        column_array = np.array(column)
        label_fit_array = label_binarizer.fit(column_array)
        for label in label_fit_array.classes_:
            if column.name in output_neuron_list:
                output_neuron_list[column.name].append(label)
            else:
                output_neuron_list[column.name] = [label]
        label_one_hot.append(label_binarizer.fit_transform(column_array))
    print(label_one_hot)

    train_Labels = []
    test_Labels = []
    for label in label_one_hot:
        split = train_test_split(data_one_hot, label, shuffle=False)
        (trainX, testX, trainLabelY, testLabelY) = split
        test_Labels.append(testLabelY)
        train_Labels.append(trainLabelY)
    print(testX)
    return trainX, testX, train_Labels, test_Labels, input_neuron_list, output_neuron_list


def data_preprocessing_predicting(pandas_data, features_data):
    Concat_Data = pandas_data.append(features_data, ignore_index=True)
    data_Array = np.array(Concat_Data)
    # data_Array = np.array(features_data)
    one_Hot = OneHotEncoder()
    data_One_Hot = one_Hot.fit_transform(data_Array).toarray()

    return data_One_Hot


def data_preprocessing_predicting_mf(pandas_data, features_data):
    Concat_Data = pandas_data.append(features_data, ignore_index=True)
    data_Array = np.array(Concat_Data)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(categorical_transformer)
        #transformers=[
            #('cat', categorical_transformer, [0])
        #])

    data_One_Hot = preprocessor.fit_transform(data_Array)

    return data_One_Hot

