import numpy as np
import pickle

# Loading the model
loaded_model=pickle.load(open('E:/DiseasePrediction/saved_models/diabetes_model.sav','rb'))


input_data=(5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array=np.array(input_data)

# Reshaping input_data_as_numpy_array
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
     print(" NOT DIABETIC")
else:
     print(" DIABETIC")