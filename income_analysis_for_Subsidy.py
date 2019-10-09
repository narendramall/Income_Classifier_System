# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:03:51 2018

@author: Narendra
"""
#first import necessary library 
#import pandas to generate and work on dataframe
import pandas as pd

#import numpy to do various calculation
import numpy as np

#import seaborn for visualisation
import seaborn as sns

#for partition the data
from sklearn.model_selection import train_test_split

#import library for logistic regression 
from sklearn.linear_model import LogisticRegression

#to look for performance importing accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

data_income=pd.read_csv('income.csv')

#creating a copy of original data
data=data_income.copy()
#
#Exploratory data Analysis:
#1. Getting to know about the data
#2. Data Preprocessing (Handling missing value)
#3. Cross table and data visualisation
#

#so start with first
#getting to know about data
data.info()

#check for missing values
data.isnull()

print('data column with null value:\n',data.isnull().sum())
#so here we got no missing values

#summary of numerical variable
summary_num=data.describe()

#summary of categorical variable
summary_cat=data.describe(include='O')

#frequency of each category
data['JobType'].value_counts()
data['occupation'].value_counts()

np.unique(data['JobType'])
np.unique(data['occupation'])
#there exist ' ?' instead of nan

#go back and read the data again by including 'na_values=[' ?']' to get ' ?' recognize as nan
data=pd.read_csv('income.csv',na_values=[' ?'])

missing = data[data.isnull().any(axis=1)]
#axis =1to consider at least one column value is missing
#will be consider as missing row
#
#Note:
#    1.missing values in jobtype= 1809
#    2.missing value in occupation=1816
#    3.there are 7 row having jobtype as never worked
#

#drop the missing row column
data2=data.dropna(axis=0)

#relationship b/w numerical independent variable
#correlation=data2.corr()

#cross table and data visualization
#extracting the column name
data2.columns

##
#gender proportion table:
gender = pd.crosstab(index=data2['gender'],
                     columns='count',
                     normalize=True)

#gender vs salery status:

gender_salstat=pd.crosstab(index=data2['gender'],
                           columns=data2['SalStat'],
                           margins=True,
                           normalize='index')


#visualization 
SalStat=sns.countplot(data2['SalStat'])
#so here we can see that almost 75% people have salery less than 50k

# histogram of age
sns.distplot(data2['age'],bins=10,kde=False)
# people with age 20-45 age are in high freq


#######################
###### Apply Logistic Regression 
#############################

# encoding salstat to 0 and 1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})

new_data=pd.get_dummies(data2,drop_first=True)

#storing the column name
columns_list=list(new_data.columns)

# seperating the input name from data
features=list(set(columns_list)-set(['SalStat']))

#storing the output values in y
y= new_data['SalStat'].values

#storing the input in x
x= new_data[features].values

#splitting the data into train and test 
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

# Make an Instance of model
logistic =LogisticRegression()

# fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#prediction from test data
prediction=logistic.predict(test_x)

#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)

#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
## here we got 84.23% accuracy

#count the misclassified result
print((test_y!=prediction).sum())
#so we got here 1427 misclassified result with the above model

####################################
### Logistic regression after removing insignificant variable
#######################################

# columns to be remove
cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data,drop_first=True)

#new column to be consider
columns_list=list(new_data.columns)
# storing the input names from data
features=list(set(columns_list)-set(['SalStat']))

#storing the output value in y
y=new_data['SalStat'].values

# storing the input variable in x
x=new_data[features].values

# splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

# Make an Instance of Model
logistic= LogisticRegression()

# Fitting the values for x, y
logistic.fit(train_x,train_y)

# prediction from test data
prediction=logistic.predict(test_x)
prediction.reshape(9049,1)

test_y.reshape(9049,1)

#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)

#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
## here we got 83.88% accuracy
# here we can see that accuracy decreased
# it is because of we left few column so we lost data



# to check accuracy of new model
print((test_y!=prediction).sum()) # will tell you misclassified ans
#so accuracy_score=(1-(test_y!=prediction).sum())/9049


#############################################
######  KNN
###########################################

#importing the library of knn
from sklearn.neighbors import KNeighborsClassifier

#import library for plotting
import matplotlib.pyplot as plt

# get the instance of KNN classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=5)

# fitting the values for x and y
KNN_classifier.fit(train_x,train_y)

#predicting the test with knn model
prediction=KNN_classifier.predict(test_x)

print((test_y!=prediction).sum()) # will tell you misclassified ans
#so here we get 83.37% accuracy

#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)

#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

#######################
#effect of K value on classifier
#########################

Misclassified_sample=[]

#calculating error for k values between 1 and 20

for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y!=pred_i).sum())
    
print(Misclassified_sample)

####################
### End of project
###########
## Logistic Reression perform better than KNN 





















