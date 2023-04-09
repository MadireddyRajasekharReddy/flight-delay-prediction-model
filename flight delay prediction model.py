#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


# In[2]:


flights = pd.read_csv('Desktop/flights.csv')
flights


# In[3]:


flights_needed_data = flights[0:100000]


# In[4]:


flights_needed_data


# In[5]:


flights_needed_data.info() 


# In[6]:


flights_needed_data.value_counts('DIVERTED')


# In[7]:


sb.jointplot(data=flights_needed_data, x="SCHEDULED_ARRIVAL", y="ARRIVAL_TIME")


# In[8]:


corr = flights_needed_data.corr(method='pearson')


# In[9]:


sb.heatmap(corr)


# In[10]:


corr


# In[11]:


flights_needed_data=flights_needed_data.drop(['YEAR','FLIGHT_NUMBER','AIRLINE','DISTANCE','TAIL_NUMBER','TAXI_OUT',
                                              'SCHEDULED_TIME','DEPARTURE_TIME','WHEELS_OFF','ELAPSED_TIME',
                                              'AIR_TIME','WHEELS_ON','DAY_OF_WEEK','TAXI_IN','CANCELLATION_REASON'],
                                             axis=1)


# In[12]:


flights_needed_data


# In[13]:


flights_needed_data=flights_needed_data.fillna(flights_needed_data.mean())


# In[14]:


flights_needed_data


# In[15]:


result=[]


# In[16]:


for row in flights_needed_data['ARRIVAL_DELAY']:
  if row > 15:
    result.append(1)
  else:
    result.append(0) 


# In[17]:


flights_needed_data['result'] = result


# In[18]:


flights_needed_data


# In[19]:


flights_needed_data.value_counts('result')


# In[20]:


flights_needed_data=flights_needed_data.drop(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'],axis=1)
flights_needed_data


# In[21]:


data = flights_needed_data.values
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 


# In[22]:


scaled_features = StandardScaler().fit_transform(X_train, X_test)


# In[23]:


clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)


# In[24]:


pred_prob = clf.predict_proba(X_test)
auc_score = roc_auc_score(y_test, pred_prob[:,1])
auc_score


# In[ ]:




