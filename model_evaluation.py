import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import csv
import os
import heapq

from tensorflow.keras.utils import plot_model


class ConLearn:

    def build_model(input_shape, label_dict):

        inputs = tf.keras.Input(shape=(input_shape,), name="Configuration_data")
        x = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu)(inputs)
        outputs = []
        for label_name, labels in label_dict.items():
            output_shape = len(labels)
            outputs.append(ConLearn.build_branch(input_shape, output_shape, x, label_name))

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="ConLearn")

        return model

    def build_branch(input_shape, output_shape, x, label_name):
        if output_shape < 3:
            x = tf.keras.layers.Dense(output_shape, activation=tf.nn.sigmoid, name=label_name)(x)
        else:
            x = tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax, name=label_name)(x)

        return x

    def model_evaluation(model, losses, lossWeights, trainX, testX, trainLabels, testLabels,
                         label_Dict, features_Dict, prediction_names, settings):
        epochs = 32
        optimizer = tf.optimizers.Adam()
        model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights,
                      metrics=["accuracy"])
        model.summary()
        if len(trainLabels) == 1:
            trainLabels = trainLabels[0]
        if len(testLabels) == 1:
            testLabels = testLabels[0]
        history = model.fit(trainX, trainLabels,
                            validation_data=(testX, testLabels),
                            epochs=epochs, batch_size=16, verbose=1,
                            label_dict=label_Dict, features_dict=features_Dict, prediction_names=prediction_names,
                            settings=settings)

        # save model
        id = str(uuid.uuid4())
        try:
            os.makedirs("Models/" + id)
        except:
            print("Directory " + "Models/" + id + " already exists!")
        model.save("Models/" + id + "/model.h5")

        # print model diagrams
        # os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'
        plot_model(model, "Models/" + id + "/model.png", show_shapes=True, show_layer_names=True)

        print('\nhistory dict:', history.history)

        history_Losses = []
        history_Accuracy = []
        for item in history.history:
            if 'val' in item:
                continue
            elif 'loss' in item:
                history_Losses.append(item)
            elif 'accuracy' in item:
                history_Accuracy.append(item)
            else:
                print('Unknown history item' + item)
        """
        plt.style.use("ggplot")

        # print loss
        (fig, ax) = plt.subplots(len(history_Losses), 1, figsize=(15, len(history_Losses) * 3))
        # loop over the loss names
        for (i, l) in enumerate(history_Losses):
            # plot the loss for both the training and validation data
            title = "Loss for {}".format(l) if l != "loss" else "Total loss"
            ax[i].set_title(title)
            ax[i].set_xlabel("Epoch #")
            ax[i].set_ylabel("Loss")
            ax[i].plot(np.arange(0, epochs), history.history[l], label=l)
            ax[i].plot(np.arange(0, epochs), history.history["val_" + l],
                       label="val_" + l)
            ax[i].legend()
        # save the losses figure
        plt.tight_layout()
        plt.savefig("Models/" + id + "/losses.png")
        plt.close()

        # print accuracy
        (fig, ax) = plt.subplots(len(history_Accuracy), 1, figsize=(15, len(history_Accuracy) * 3))
        # loop over the loss names
        for (i, l) in enumerate(history_Accuracy):
            # plot the loss for both the training and validation data
            title = "Accuracy of {}".format(l) if l != "accuracy" else "Total accuracy"
            ax[i].set_title(title)
            ax[i].set_xlabel("Epoch #")
            ax[i].set_ylabel("Accuracy")
            ax[i].plot(np.arange(0, epochs), history.history[l], label=l)
            ax[i].plot(np.arange(0, epochs), history.history["val_" + l],
                       label="val_" + l)
            ax[i].legend()
        # save the losses figure
        plt.tight_layout()
        plt.savefig("Models/" + id + "/accuracy.png")
        plt.close()
        """
        return id

    def save_model_csv(id, training_file_path, label_names, model_library_file_path):
        pandas_Data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
        with open(model_library_file_path, "a+", newline='') as model_Library:
            if os.stat(model_library_file_path).st_size == 0:
                head_Data = ['ID']
                for column in pandas_Data.columns:
                    head_Data.append(column)
                library_Entry = [id]
                for column in pandas_Data.columns:
                    if column in label_names:
                        library_Entry.append(1)
                    else:
                        library_Entry.append(0)
                writer = csv.writer(model_Library, delimiter=';')
                writer.writerow(head_Data)
                writer.writerow(library_Entry)
            else:
                library_Entry = [id]
                for column in pandas_Data.columns:
                    if column in label_names:
                        library_Entry.append(1)
                    else:
                        library_Entry.append(0)
                writer = csv.writer(model_Library, delimiter=';')
                writer.writerow(library_Entry)

        return

    def model_exists(model_library_file_path, label_names):
        id = str()
        try:
            Library_Data = pd.read_csv(model_library_file_path, delimiter=';')
            model_exists = 0
            for i in range(Library_Data.shape[0]):
                for element in Library_Data:
                    if element in label_names:
                        if Library_Data[element][i] == 1:
                            model_exists = 1
                            continue
                        else:
                            model_exists = 0
                            break
                    else:
                        if Library_Data[element][i] == 1:
                            model_exists = 0
                            break
                        else:
                            continue
                if model_exists:
                    id = Library_Data['ID'][i]
                    break
            return id
        except:
            return id

    def model_predict(id, test_input, label_names, label_dict, prediction_rest_file_path):
        model = tf.keras.models.load_model("Models/" + id + "/model.h5")
        predictions = model.predict(test_input)  # last predicted values are the desired ones

        item_predictions = []
        if type(predictions) is list:
            for pred in predictions:
                item_predictions.append(pred[len(pred) - 1])
        else:
            item_predictions.append(predictions[len(predictions) - 1])

        pred_dict = {}
        for i in range(len(label_names)):
            pred_dict[label_names[i]] = item_predictions[i]

        prediction_accuracy = {}
        for label_Name in label_names:
            prediction_accuracy[label_Name] = max(pred_dict[label_Name])

        if len(prediction_accuracy) > 1:
            prediction_top = heapq.nlargest(1, prediction_accuracy, key=prediction_accuracy.get)
        else:
            prediction_top = prediction_accuracy  # last prediction has to have more than one feature

        prediction_values = {}
        prediction_rest = {}

        for label_Name in label_names:
            if label_Name in prediction_top:
                for i in range(len(pred_dict[label_Name])):
                    if pred_dict[label_Name][i] == max(pred_dict[label_Name]):
                        # index = np.where(pred_dict[label_Name] == max(pred_dict[label_Name]))
                        # predict value with highest probability
                        prediction_values[label_Name] = label_dict[label_Name][i]
                    else:
                        prediction_rest[label_dict[label_Name][i]] = pred_dict[label_Name][i]

        dict(sorted(prediction_rest.items(), key=lambda item: item[1]))
        # save sorted predictions to file to apply if prediction is invalid
        with open(prediction_rest_file_path, "w", newline='') as rest:
            head_data = ['Property']
            if type(prediction_top) is list:
                probability = [prediction_top[0]]
            else:
                probability = [list(prediction_top.keys())[0]]
            for item in prediction_rest:
                head_data.append(item)
            for item in prediction_rest:
                probability.append(prediction_rest[item])
            writer = csv.writer(rest, delimiter=';')
            writer.writerow(head_data)
            writer.writerow(probability)

        print(prediction_values)
        return prediction_values

    def model_predict_cluster(id, test_input, label_names, label_dict, prediction_names, prediction_rest_file_path):
        model = tf.keras.models.load_model("Models/" + id + "/model.h5")
        predictions = model.predict(test_input)  # last predicted values are the desired ones

        item_predictions = []

        for pred in predictions:
            item_predictions.append(pred[len(pred) - 1])

        pred_dict = {}
        for i in range(len(label_names)):
            for j in range(len(prediction_names)):
                if label_names[i] == prediction_names[j]:
                    pred_dict[prediction_names[j]] = item_predictions[i]

        prediction_accuracy = {}
        for prediction_Name in prediction_names:
            prediction_accuracy[prediction_Name] = max(pred_dict[prediction_Name])

        prediction_top = heapq.nlargest(1, prediction_accuracy, key=prediction_accuracy.get)

        prediction_values = {}
        prediction_rest = {}

        for prediction_Name in prediction_names:
            if prediction_Name in prediction_top:
                for i in range(len(pred_dict[prediction_Name])):
                    if pred_dict[prediction_Name][i] == max(pred_dict[prediction_Name]):
                        # index = np.where(pred_dict[label_Name] == max(pred_dict[label_Name]))
                        # predict value with highest probability
                        prediction_values[prediction_Name] = label_dict[prediction_Name][i]
                    else:
                        prediction_rest[label_dict[prediction_Name][i]] = pred_dict[prediction_Name][i]

        dict(sorted(prediction_rest.items(), key=lambda item: item[1]))
        # save sorted predictions to file to apply if prediction is invalid
        with open(prediction_rest_file_path, "w", newline='') as rest:
            head_data = ['Property']
            if type(prediction_top) is list:
                probability = [prediction_top[0]]
            else:
                probability = [list(prediction_top.keys())[0]]
            for item in prediction_rest:
                head_data.append(item)
            for item in prediction_rest:
                probability.append(prediction_rest[item])
            writer = csv.writer(rest, delimiter=';')
            writer.writerow(head_data)
            writer.writerow(probability)

        print(prediction_values)
        return prediction_values
