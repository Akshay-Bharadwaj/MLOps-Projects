# Create a model that predicts profit of company based on spending pattern and company's location
# SL = 0.1
# Deploy model once created
     

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle


data = pd.read_csv("50_Startups.csv")

#Seperate data as features and label
features = data.iloc[:,[0,1,2,3]].values
label = data.iloc[:,[4]].values
     
# Since, categorical feature is present, let us do OHE (One Hot Encoding)
oheState = OneHotEncoder(sparse_output=False)
stateDummy = oheState.fit_transform(data.iloc[:, [3]])

# concatenate the encoded state with the original dataframe
finalFeatureSet = np.concatenate((stateDummy,features[:,[0,1,2]]) , axis = 1)
finalFeatureSet

# Model train
X_train, X_test, y_train, y_test = train_test_split(finalFeatureSet, label, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)
     
# Deploying model
pickle.dump(model, open("ProfitPredictor.pkl", "wb"))
pickle.dump(oheState, open("StateConverter.obj", "wb"))

print("✅ Model is saved.")
