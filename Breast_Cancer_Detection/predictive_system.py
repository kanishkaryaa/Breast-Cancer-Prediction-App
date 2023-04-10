import numpy as np
import pickle

# Loading the saved model

filename = 'breast_cancer_prediction_model.sav'
loaded_model = pickle.load(open(filename,'rb'))

input_data = (19.81,22.15,130,1260,0.09831,0.1027,0.1479,0.09498,0.1582,0.05395,0.7582,1.017,5.865,112.4,0.006494,0.01893,0.03391,0.01521,0.01356,0.001997,27.32,30.88,186.8,2398,0.1512,0.315,0.5372,0.2388,0.2768,0.07615)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are prediction for one data point

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print("The Breast cancer is Malignant")
    
else:
    print("The Breast Cancer is Benign")