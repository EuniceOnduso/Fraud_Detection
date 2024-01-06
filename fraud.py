# %%
from sklearn.model_selection import train_test_split as ts
from sklearn.linear_model import LogisticRegression #the model we will use
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score #checks accuracy of our model
import joblib

# %%
#read the csv file
csv = pd.read_csv('../archive/creditcard.csv')

# %%
#read the first 5 rows of the data to see what is contained in the csv
csv.head()

# %%
#checking the last five transactions
csv.tail()
#Time is the seconds of transaction after first transaction
#class 0 means fraudulent, 1 means non-fraudulent

# %%
#checking the information of the dataset
csv.info()
#we have gotten the info, how many nulls and the data types

# %%
#can check for nulls
total_null = csv.isnull().sum().sum()
print("Total Null is: ")
print(total_null)

# %%
#checking the distribution of the 0s and 1s
csv['Class'].value_counts()
#we have gotten the distribution of both the good and bad transactions
#the distribution is unbalanced and cannot train these data to the model
#if we do so, any transaction done might come out non-fraudulent

# %%
#seperate the data of 0s and 1s
fraud = csv[csv.Class == 1]
legit = csv[csv.Class == 0]

# %%
#get the shape, rows and cols
print(fraud.shape)
print(legit.shape)

# %%
#statistical data
legit.Amount.describe()

# %%
fraud.Amount.describe()
#the mean of the fraud is way larger that the mean of legit

# %%
#we compare both the means
csv.groupby('Class').mean()

# %%
#Build a similar dataset of the legit and fraudulent transaction to help in training
#this will make the data set uniform and balanced
fraud_count = len(csv[csv['Class'] == 1])
print(fraud_count)

# %%
#we get a random sample of non-fraudulent
legit_sample = legit.sample(n = fraud_count)

# %%
#we concatnate the 2 data frames
new_dataset = pd.concat([legit_sample, fraud], axis=0)
#axis 0 mean rows axis 1 mean cols

# %%
new_dataset.head()

# %%
new_dataset.tail()

# %%
#get the new count
new_dataset.Class.value_counts()

# %%
#compare with the new balanced dataset to see the differences
new_dataset.groupby('Class').mean()
#the new difference shows that the nature of the two does not change, the fraud is still more

# %%
#splitting the data set into features and targets
#features are the data that contribute to the decision of whether a transaction is fraud or legit
#targets is data that we aim to attain with the features
x = new_dataset.drop(columns='Class', axis=1)#remove target we get features
y = new_dataset['Class']#target

# %%
print(x)

# %%
print(y)

# %%
#we need to split the data into training and testing data
x_train, x_test, y_train, y_test = ts(x,y,test_size=0.2,stratify=y, random_state=2)
#the train test split randomly splits the data into training set and testing set and store them into corresponding variables
# test size just means how much data to be used in test group
#stratify just shows which data set is the creteria for the random split
#random state is how many time you want to split this data

# %%
print(x.shape, x_train.shape, x_test.shape)

# %%
model = LogisticRegression()
#loading model

# %%
#training the model
model.fit(x_train, y_train)

# %%
#evaluate the model according to the accuracy score
#accuracy on training data
x_train_prediction = model.predict(x_train)
x_train_score = accuracy_score(x_train_prediction, y_train)
#print the score
print(x_train_score)

# %%
#accuracy on test data
x_test_prediction = model.predict(x_test)
x_test_score = accuracy_score(x_test_prediction, y_test)
print(x_test_score)

# %%
# print("in the new prediction")
# def myPrediction(data_frame: pd.DataFrame):
#     new_data_frame = data_frame.drop(columnc = 'Class', axis=1)
#     return model.predict(new_data_frame)
    
csv2 = pd.read_csv('../archive/Book1.csv')
print(csv2.to_json(orient='records'))


# # %%
# csv2.shape

# # %%
# x2 = csv2.drop(columns='Class', axis=1)

# # %%
# result = model.predict(x2)
# print("the result is: %s",result)

joblib.dump(model, "mymodel.pkl")