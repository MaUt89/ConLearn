# ConLearn
ConLearn is a machine learning programm able to learn and predict configurations based on a historical configurations (training data set). 
The variable value ordering branch integrates the prediction made by ConLearn into the search heuristics of a constraint solver.

# learn.py
Starting method for intilizing learning of the model. Label names, feature data and labels to be predicted are determined on basis of a csv file specifying configuration phases of the configuration process.
Training data is prepared and the machine learning model is initalized. The model is trained based on the training data and the model is saved to a repository. The ID and the specification of the model is saved to a model library. This procedure is repeated as often as specified in the input parameter models_to_learn. If more than one model is learned the model achieving the best prediction results is maintained.
After learning the model the algorithm directly utilizes the model to predict configurations which can be done in four different scenarios:
1. Directly using the predictions of the ML model as input for the configurator
2. Encoding the preidctions of the ML model as variable value orderings for the constraint solver of the configurator
3. Directly using the predictions of a nearest neighbor approach
4. Encoding the predictions of a nearest neighbor approach

# model_evaluation.py
build_model
Algorithm is used to create the neural network used for ML.

model_evaluation
Algorithm is utilized to learn the ML models based on the tensorflow model.fit() method.

model_predict_xxx
After the training session of the ML model this method is called to determine the performance of the model by checking the consistency and accuracy of the prediction based on the validation data. There exist four different methods to perform the prediction:
1. model_predict: Simply predicts the one variable value to be predicted based on the ML model
2. model_predict_cluster: Preidicts all variable values of the variables within the predefined cluster of variables based on the ML model
3. model_predict_choco: Predicts all variable values of the variables within the predefined cluster and creates input for the choco solver
4. model_predict_solver: Predicts all variable values of the variables within the predefined cluster, stores the variable value ordering per variable and provides this as an input for a constraint solver

# validation_choco
Validates the consistency and accuracy of the predictions made by calling the constraint solver of the configurator providing the predictions made by the ML model.

# nearest_neighbor.py
A second methodology to predict configurations based on the nearest neighbor approach to validate the performance of the machine learning model to a muchg simpler approach.

# XML_handling
Algorithm reads the input XML that determines the requirement which variables of the configuration are missing and is therefore the basis to determine which variables should be predicted by the ML model.

# data_handling
Algorithm prepares the input for the ML by preprocessing the training data and by determining the labeled data.

# Training data switchgear
Example training data anonymized from real life example of a productive industrial switchgear configurator.
