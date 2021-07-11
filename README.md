# ConLearn
ConLearn is a machine learning programm able to learn and predict configurations based on a historical configurations (training data set). 
It is also capable of semantic regularization during training phase by penalizing wrong predictions during learning phase.  

# main.py
Flask based webservice communicating with the configurator software to predict unselected variables of the current configuration.

# learn.py
Starting method for intilizing learning of the model. Label names, feature data and labels to be predicted are dfetermined on basis of a csv file specifying configuration phases of the configuration process.
Training data is prepared and the machine learning model is initilized. The model is trained based on the training data and the model is saved to a repository. The ID and the specification of the model is saved to a model library.

# evaluate.py
During training session of the tensorflow model.fit method this method is calleed during each evaluation step of the model training. For each prediction during the evaluation session an XML-file is created and validated by the configurator, whether the prediction was correct or not. 
If it was not the current model is penalized by increasing the loss determined for the current model and forcing in this manor a bigger adjustment in the next evaluation period.

# prediction.py
Method is utilizing the model library to predict configurations. First, labels, features data and labels to be predicted are recognized based on a request XML-file similarly to the learning phase. 
The algorithm checks, whether a previously predicted variable is still part of the request for prediction and sets the next propable value based on the previous prediction. 
If no previously predicted variable is part of the request it is checked if a model has been learned to predict the current request. The model is loaded from the library if this can be answered positively.
The prediction is performed and an answer XML-file is created.

# prediction_neighbor.py
A second methodology to predict conbfigurations based on the nearest neighbor approach to validate the performance of the machine learning model to a muchg simpler approach.

