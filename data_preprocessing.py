from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def data_preprocessing_learning(pandas_data, label_columns):
    data_Array = np.array(pandas_data)
    one_Hot = OneHotEncoder()
    data_One_Hot = one_Hot.fit_transform(data_Array).toarray()
    print(data_One_Hot)
    label_one_hot = []
    label_binarizer = LabelBinarizer()

    for column in label_columns:
        column_array = np.array(column)
        label_one_hot.append(label_binarizer.fit_transform(column_array))
    print(label_one_hot)

    train_Labels = []
    test_Labels = []
    for label in label_one_hot:
        split = train_test_split(data_One_Hot, label, shuffle=False)
        (trainX, testX, trainLabelY, testLabelY) = split
        test_Labels.append(testLabelY)
        train_Labels.append(trainLabelY)
    print(testX)
    return trainX, testX, train_Labels, test_Labels


def data_preprocessing_predicting(pandas_data, features_data):
    Concat_Data = pandas_data.append(features_data, ignore_index=True)
    data_Array = np.array(Concat_Data)
    # data_Array = np.array(features_data)
    one_Hot = OneHotEncoder()
    data_One_Hot = one_Hot.fit_transform(data_Array).toarray()

    return data_One_Hot
