import pickle  
import pandas as pd  
import numpy as np  

# Load the trained model from the pickle file  
with open('pickle_shapelet_model.pkl', 'rb') as file:  
    model = pickle.load(file)  
  
# Load the test data    
test_data = pd.read_excel('Test_3.xlsx')   
X_test= test_data['Faulted rotor bearing'].values
X1=[]
ub=512
lb=0
while ub<=4096:
      X1.append(X_test[lb:ub])
      lb = ub
      ub = ub +512
# X1.append(X_test[:])
# print (X1)
X_test =np.array(X1)

y_Test= "Faulted rotor bearing"

# Make predictions    
# print(X_test[0,:5])
predictions = model.predict(X_test)   

print(predictions)