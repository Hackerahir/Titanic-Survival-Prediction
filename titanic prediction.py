#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


df1=pd.read_csv('titanic_train.csv') 


# In[5]:


df1.head()


# In[6]:


df1.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis=1 , inplace=True)


# In[7]:


df1.head()


# In[8]:


df1.isnull().sum()


# In[9]:


df1['Age'].describe()


# 

# In[10]:


df1['Age'].fillna(df1['Age'].mean(),inplace=True)


# In[11]:


df1.isnull().sum()


# In[15]:


l_sex_dummies=pd.get_dummies(df1['Sex'],drop_first=True)


# In[16]:


df1=pd.concat([df1,l_sex_dummies],axis=1)


# In[17]:


df1.head()


# In[18]:


df1.drop(['Sex'],axis=1,inplace=True)


# In[19]:


df1.head()


# In[27]:


from sklearn.preprocessing import StandardScaler
sts=StandardScaler()


# In[29]:


feature_scale=['Age','Fare']
df1[feature_scale]=sts.fit_transform(df1[feature_scale])


# In[30]:


df1.head()


# In[31]:


x=df1.drop(['Survived'],axis=1)
y=df1['Survived']


# In[35]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[36]:


model_param = {
    'DecisionTreeClassifier':{
        'model':DecisionTreeClassifier(),
        'param':{
            'criterion': ['gini','entropy']
        }
    },
        'KNeighborsClassifier':{
        'model':KNeighborsClassifier(),
        'param':{
            'n_neighbors': [5,10,15,20,25]
        }
    },
        'SVC':{
        'model':SVC(),
        'param':{
            'kernel':['rbf','linear','sigmoid'],
            'C': [0.1, 1, 10, 100]
         
        }
    }
}


# In[38]:


scores =[]
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)
    model_selection.fit(x,y)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })


# In[39]:


df_model_score = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df_model_score


# In[40]:


model_svc = SVC( C= 100,kernel='rbf')


# In[41]:


model_svc.fit(x, y)


# In[59]:


df4 = pd.read_csv('titanic_test.csv')


# In[60]:


df4.head()


# In[61]:


df3=df4.drop(['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'], axis=1 )


# In[62]:


df3.head()


# In[63]:


df3.isnull().sum()


# In[64]:


df3['Age'].fillna(df3['Age'].mean(),inplace=True)
df3['Fare'].fillna(df3['Fare'].mean(),inplace=True)


# In[65]:


df2.isnull().sum()


# In[67]:


l_sex_dummies=pd.get_dummies(df3['Sex'],drop_first=True)
df3= pd.concat([df3,l_sex_dummies],axis=1)
df3.drop(['Sex'], axis=1, inplace=True )


# 

# In[68]:


df3.head()


# 

# In[69]:


df3[feature_scale] = sts.fit_transform(df3[feature_scale])


# df3.head()

# In[70]:


df3.head()


# In[71]:


y_predicted = model_svc.predict(df3)


# In[73]:


submission = pd.DataFrame({
        "PassengerId": df4['PassengerId'],
        "Survived": y_predicted
    })


# In[74]:


submission.to_csv('titanic_output.csv', index=False)


# In[ ]:




