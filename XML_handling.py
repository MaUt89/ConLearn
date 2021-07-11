import pandas as pd
import xml.etree.ElementTree as et
import os
import collections


def xml_parse(input_xml, irrelevant_features):
    label_names = []
    feature_columns = []
    features = {}
    irrelevant = False
    tree = et.parse(input_xml)
    root = tree.getroot()

    for items in root:
        for item in items:
            with open(irrelevant_features, "r") as irr_Features:
                for lines in irr_Features:
                    if lines.strip() == item.attrib['key']:
                        irrelevant = True
            if not irrelevant and item.attrib['key'] not in feature_columns:
                if not item.attrib['value'] or item.attrib['valid'] == '0':
                    label_names.append(item.attrib['key'])
                else:
                    feature_columns.append(item.attrib['key'])
                    features[item.attrib['key']] = item.attrib['value']
            irrelevant = False

    features_data = pd.DataFrame(features, columns=feature_columns, index=[0])
    return label_names, features_data


def xml_parse_categories(input_xml, irrelevant_features, categories_csv):
    label_names = []
    prediction_names = []
    prediction_candidate = []
    feature_columns = []
    features = {}
    category_dict = {}
    count = 0
    position = 0
    irrelevant = False
    categories_Data = pd.read_csv(categories_csv, delimiter=';', dtype='string')
    tree = et.parse(input_xml)
    root = tree.getroot()

    for items in root:
        for item in items:
            with open(irrelevant_features, "r") as irr_Features:
                for lines in irr_Features:
                    if lines.strip() == item.attrib['key']:
                        irrelevant = True
            if not irrelevant:
                if item.attrib['key'] not in label_names:
                    if not item.attrib['value'] or item.attrib['valid'] == '0':
                        for category in categories_Data:
                            if item.attrib['key'] in categories_Data[category].unique():
                                for label in categories_Data[category]:
                                    if not label == 'empty':
                                        if not category_dict:
                                            label_names.append(label)
                                        else:
                                            for cat in category_dict:
                                                if category < cat:
                                                    label_names.insert(position + count, label)
                                                    break
                                                else:
                                                    if count == 0:
                                                        position += category_dict[cat]
                                                    if cat == list(category_dict)[-1]:
                                                        label_names.insert(position + count, label)
                                                        break
                                        count += 1
                                        if label in feature_columns:
                                            feature_columns.remove(label)
                                            del features[label]
                                    else:
                                        break
                                category_dict[category] = count
                                category_dict = collections.OrderedDict(sorted(category_dict.items()))
                                count = 0
                                position = 0
                                prediction_candidate.append(item.attrib['key'])
                                break
                    else:
                        if item.attrib['key'] not in feature_columns:
                            feature_columns.append(item.attrib['key'])
                            features[item.attrib['key']] = item.attrib['value']
                else:
                    if not item.attrib['value'] or item.attrib['valid'] == '0':
                        prediction_candidate.append(item.attrib['key'])
            irrelevant = False

    if category_dict:
        for i in range(list(category_dict.values())[0]):
            if label_names[i] in prediction_candidate:
                prediction_names.append(label_names[i])

    features_data = pd.DataFrame(features, columns=feature_columns, index=[0])
    return label_names, features_data, prediction_names


def xml_parse_compare(input_xml, compare_xml, irrelevant_features):
    invalid_features = []
    valid_features = []
    features = {}
    irrelevant = False
    tree = et.parse(input_xml)
    root = tree.getroot()

    for items in root:
        for item in items:
            with open(irrelevant_features, "r") as irr_Features:
                for lines in irr_Features:
                    if lines.strip() == item.attrib['key']:
                        irrelevant = True
            if not irrelevant and item.attrib['key'] not in valid_features:
                if not item.attrib['value'] or item.attrib['valid'] == '0':
                    invalid_features.append(item.attrib['key'])
                    features[item.attrib['key']] = item.attrib['value']
                else:
                    valid_features.append(item.attrib['key'])
                    features[item.attrib['key']] = item.attrib['value']
            irrelevant = False

    tree = et.parse(compare_xml)
    root = tree.getroot()

    for items in root:
        for item in items:
            with open(irrelevant_features, "r") as irr_Features:
                for lines in irr_Features:
                    if lines.strip() == item.attrib['key']:
                        irrelevant = True
            if not irrelevant and item.attrib['value'] != features[item.attrib['key']] \
                    and item.attrib['key'] in valid_features:
                # Knowledge base has assigned a new valid value because prediction was wrong
                invalid_features.append(item.attrib['key'])

            irrelevant = False
        break

    return invalid_features


def xml_write(prediction_values, progress_xml, output_xml):
    tree = et.parse(progress_xml)
    root = tree.getroot()

    for items in root:
        for item in items:
            if item.attrib['key'] in prediction_values:
                item.attrib['value'] = prediction_values[item.attrib['key']]

    tree.write(output_xml)
    os.remove(progress_xml)

    return


def training_xml_write(prediction_values, feature_values, progress_xml, evaluate_xml):
    tree = et.parse(progress_xml)
    root = tree.getroot()

    for items in root:
        for item in items:
            if item.attrib['key'] in feature_values:
                item.attrib['value'] = feature_values[item.attrib['key']]
                continue
            if item.attrib['key'] in prediction_values:
                item.attrib['value'] = prediction_values[item.attrib['key']]

    tree.write(evaluate_xml)

    return
