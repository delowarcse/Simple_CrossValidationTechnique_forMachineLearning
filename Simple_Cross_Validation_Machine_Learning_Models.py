#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')

# Load and merge datasets 
stroke_data = pd.read_csv('Stroke/Injure Participant Data.csv', delim_whitespace=False)
control_data = pd.read_csv('Healthy Control Participants Data.csv', delim_whitespace=False)

# store wine type as an attribute
stroke_data['data_type'] = 'stroke'   
control_data['data_type'] = 'control'

# merge control and stroke data
datas = pd.concat([stroke_data, control_data])
datas = datas.sample(frac=1, random_state=42).reset_index(drop=True)

# understand dataset features and values
#datas.head()

# Prepare Training and Testing Datasets
stp_features = datas.iloc[:,:-1]
stp_feature_names = stp_features.columns
stp_class_labels = np.array(datas['data_type'])

stp_train_X, stp_test_X, stp_train_y, stp_test_y = train_test_split(stp_features, stp_class_labels, 
                                                                    test_size=0.3, random_state=42)

# Feature Scaling
# Define the scaler 
stp_ss = StandardScaler().fit(stp_train_X)

# Scale the train set
stp_train_SX = stp_ss.transform(stp_train_X)

# Scale the test set
stp_test_SX = stp_ss.transform(stp_test_X)
#print(Counter(stp_train_y), Counter(stp_test_y))
#print('Features:', list(stp_feature_names))


# In[43]:


# Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from numpy import mean, std

# Build default SVM Model
#clf = SVC(random_state=42)
clf = SVC()

# evaluate model
scores_svm = cross_val_score(clf,stp_train_SX, stp_train_y, cv=10)
print(scores)

# mean and std
print('Accuracy: %f (%f)'%(mean(scores_svm),std(scores_svm)))


# In[40]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import mean, std

# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# create model
model = LogisticRegression()
# evaluate mode
scores_lr = cross_val_score(model, stp_train_SX, stp_train_y, cv=cv)
print(scores_lr)
# mean and std of accuracy
print('Accuracy: %f (%f)'%(mean(scores_lr), std(scores_lr)))


# In[46]:


from sklearn.ensemble import RandomForestClassifier

#k-folded
cv_rf = KFold(n_splits=10, random_state=1, shuffle=True)

# train the model
model_rf = RandomForestClassifier()
scores_rf = cross_val_score(model_rf, stp_train_SX, stp_train_y, cv=cv_rf)
print(scores_rf)
# mean and std of accuracy
print('Accuracy: %f (%f)'%(mean(scores_rf), std(scores_rf)))


# In[54]:


#k-folded
cv_rft = KFold(n_splits=10,  shuffle=True, random_state=1)

# train the model
model_rft = RandomForestClassifier(n_estimators=200, max_features='auto')
scores_rft = cross_val_score(model_rft, stp_train_SX, stp_train_y, cv=cv_rft) #random_state=42
print(scores_rft)
# mean and std of accuracy
print('Accuracy: %f (%f)'%(mean(scores_rft), std(scores_rft)))


# In[57]:


from sklearn.tree import DecisionTreeClassifier

#k-fold
cv_dt = KFold(n_splits=10, random_state=1, shuffle=True)
#train model
model_dt = DecisionTreeClassifier(max_depth=4)
scores_dt = cross_val_score(model_dt, stp_train_SX, stp_train_y, cv=cv_dt)

#print score
print(scores_dt)
# mean and std of accuracy
print('Accuracy (mean, std): %f (%f)'%(mean(scores_dt),std(scores_dt)))


# In[ ]:




