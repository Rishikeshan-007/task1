#!/usr/bin/env python
# coding: utf-8

# # TASK - 1

# # Predicting Percentage Of A Student Based On No.of Study Hrs....
#                                                  - By SPARKS FOUNDATION.
# **This project is done by RISHIKESHAN VEERAVELU (Datascience and business analytics intern in SPARKS Foundation)** 
# objectives :
# 
#   ● To Predict the percentage of an student based on the no. of study hours..
#   ● What will be predicted score if a student studies for 9.25 hrs/ day? 
# 

# # Method used to handle this task :

# We need to predict scores of students based on number of hrs they study as whole.This model is build using
# **linear Regression**. In this project we briefly discuss about this project

# # Simple Linear Regression :
 Shortly, we say Simple Linear Regression  is used to estimate relationship between two quantitive variables..
 Basically,simple linear regression is to know the strongness of relationhip between two variables.
 
# # Importing Libraries :
# 

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing dataset :

# In[32]:


#from pandas we import our dataset

data = pd.read_csv('C:/users/datasets/prediction.csv')
print(data)


# # Analaysis in Data :

# In[33]:


shape = data.shape    
print(shape)


# In[34]:


datatype = data.dtypes
print(datatype)


# In[35]:


data.isna().sum()


# In[36]:


data.isnull() #to check the NULL values in our data


# # Visualisation of data :

# In[37]:


data.plot(x='Hours',y='Scores',style='o')
sns.set_style('whitegrid')
plt.title("hours vs percentage")
plt.xlabel("Hours studied")
plt.ylabel("percentage score")
plt.show()


# In[38]:


data.plot(kind='bar',figsize=(9,5))
sns.set_style('whitegrid')
plt.xlabel('hours studied')
plt.ylabel('percentage score')
plt.title('hours vs percentage')
plt.show()


# # Note on this plot:

# 1. From this plot we are going to make **simple linear Regression** between hours and scores.
# 2. Here hours and scores are 2 quantative variables.

# # Data Preprocessing :

# In[39]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[40]:


print(x)


# In[41]:


print(y)


# # Splitting our dataset

# #using Skikitlearn and using train_test_split
# 
# **train_test_split** - this function used for splitting data arrays to train data and for testing data

# In[42]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# # Training our Data :

# In[44]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# our data is trained according to **Linear Regression**

# # Predicting Trained test set results :

# In[45]:


y_pred = model.predict(x_test)
print(y_pred)


# # Visualisation of test results:

# In[46]:


rd = model.coef_*x+model.intercept_
plt.scatter(x,y)
plt.plot(x,rd,color= 'g')
sns.set_style('whitegrid')
plt.xlabel("scores")
plt.ylabel("percentage")
plt.show()


# # To predict Percentage of a student based on study hours :

# In[47]:


hours = 9.25
task = model.predict([[hours]])
print("No.of hrs :{}".format(hours))
print("predicted percentage:{}".format(task[0]))


# # Result :

# **As per our model prediction if a student read 9.25 hours per day by our prediction approximately he can surely get 93.89% marks in his exams..**

# # Thank you !!!!

# In[ ]:




