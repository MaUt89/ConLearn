# ConLearn
ConLearn is a machine learning programm able to learn and predict configurations based on a historical configurations (training data set). 
The configuration create branch can be used to augment additional knowledge base consistent training data .

# configuration_create.py
Starting method for intilizing creation of configurations. Range of the first for-loop determines how many configurations should be created. Algorithm determines the variable value distribution for each variable within the existing training dataset. A configuration out of the ecisting training dataset is randomly selected. The algorithm iterates through all variables and changes the variable value with a probability of 5%.The resulting random configuration is validated and if consistent added to the training dataset. iuf inconsistent the configuration is nglected and the next iteration starts.

# validation_choco
Validates the consistency and accuracy of the predictions made by calling the constraint solver of the configurator providing the predictions made by the ML model.

# XML_handling
Algorithm reads the input XML that determines the requirement which variables of the configuration are missing and is therefore the basis to determine which variables should be predicted by the ML model.

# data_preperation
Extends the training data file with the valid configurations created during the configuration_create method.

# Training data switchgear
Example training data anonymized from real life example of a productive industrial switchgear configurator.
