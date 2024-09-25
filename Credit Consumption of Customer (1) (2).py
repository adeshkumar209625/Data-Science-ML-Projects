#!/usr/bin/env python
# coding: utf-8

# ## Predict Credit Consumption of Customer For Leading Bank

# ## Business Problem:
# ##### Analytics driving every industry based on a variety of technology platforms which collect information from various sources by analysing what customers certainly want. The Credit Card industry is also data rich industry and data can be leveraged in infinite ways to understand customer behaviour.
# 
# ##### The data from a credit card processor shows the consumer types and their business spending behaviours. Therefore, companies can develop the marketing campaigns that directly address consumers’ behaviour. In return, this helps to make better sales and the revenue undoubtedly grows greater sales.
# 
# #### Understanding the consumption pattern for credit cards at an individual consumer level is important for customer relationship management. This understanding allows banks to customize for consumers and make strategic marketing plans. Thus it is imperative to study the relationship between the characteristics of the consumers and their consumption patterns.

# ### Importing libraries 

# In[207]:


#Packages related to data importing, manipulation, exploratory data analysis, data understanding
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV,cross_val_score

from sklearn import metrics
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor 

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler





# ### Loading Datasets

# In[61]:


os.chdir('C:\\Users\\Adesh mishra\\OneDrive\\Desktop\\ML  Case study\\New folder')


# In[62]:


cons = pd.read_excel('CreditConsumptionData.xlsx')
cust = pd.read_excel('CustomerBehaviorData.xlsx')
demo = pd.read_excel('CustomerDemographics.xlsx')


# #### Data preparation and Cleaning

# In[63]:


cons.head()#/ tail


# In[64]:


cust.head()#/ tail


# In[65]:


demo.head()#/ tail


# In[66]:


#merging all datasets 
key_var = 'ID'
dev_df = pd.merge(demo,pd.merge(cust,cons,on = key_var),on = key_var)


# In[67]:


dev_df.head()


# In[68]:


dev_df.info()


# In[69]:


dev_df.drop('ID',inplace = True,axis = 1)


# In[70]:


dev_df.columns


# In[71]:


#summary
dev_df.describe()


# In[72]:


dev_df.shape


# In[74]:


dev_df.duplicated().sum()


# In[75]:


dev_df.isnull().sum()


# In[76]:


#filling null values through mean()
dev_df.fillna(dev_df.mean(),inplace=True)


# In[77]:


dev_df.dropna(inplace= True)


# In[78]:


dev_df.isnull().sum()


# In[79]:


#checking unique values in objec cols.
for cols in dev_df.describe(include="object").columns:
    print(cols)
    print(dev_df[cols].unique())
    print("-"*100)


# In[80]:


#Removing nan valuess from objects cols..
for cols in dev_df.describe(include="object").columns:
    mode_value = dev_df[cols].mode()[0]
    dev_df[cols].fillna(mode_value,inplace=True)


# In[81]:


for cols in dev_df.describe(include="object").columns:
    print(cols)
    print(dev_df[cols].unique())
    print("-"*100)


# In[82]:


#separating numeric and categorical cols.
dev_df_conti= dev_df.select_dtypes(['int','float'])
dev_df_cat = dev_df.select_dtypes(['object'])


# In[83]:


dev_df_conti.head()


# In[84]:


dev_df_conti.describe()


# In[85]:


def continuous_var_summary( x ):
     return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(),
                       x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),
                       x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75),
                       x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

conti_summary=dev_df_conti.apply(lambda x: continuous_var_summary(x))
conti_summary


# In[86]:


def cat_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.value_counts(),x.nunique()], 
                  index=['N', 'NMISS', 'ColumnsNames','Cardinality'])
cat_summary=dev_df_cat.apply(lambda x: cat_summary(x))
cat_summary


# In[87]:


#Outliers Treatment...
dev_df_conti=dev_df_conti.apply( lambda x: x.clip(lower = x.quantile(0.01), 
                                                                upper = x.quantile(0.99)) )


# In[88]:


for i in dev_df_conti:
    plt.figure(figsize=(3,1))
    sns.boxplot(data=dev_df_conti,x=i)
    plt.show()


# In[89]:


##outlier treatment

def treat_outlier(x):
    q1=x.quantile(0.25)
    q3=x.quantile(0.75)
    iqr=q3-q1
    lr=q1-1.5*iqr
    ur=q3+1.5*iqr
    return x.clip(lower=lr,upper=ur)


# In[90]:


for column in dev_df_conti.columns:
    dev_df_conti[column]=treat_outlier(dev_df_conti[column])


# In[91]:


for i in dev_df_conti:
    plt.figure(figsize=(3,1))
    sns.boxplot(data=dev_df_conti,x=i)
    plt.show()


# In[92]:


dev_df_conti


# In[97]:


#putting conti columns in a new var and removing target var..
cols = list(dev_df_conti.columns)
cols.remove("cc_cons")


# In[98]:


for i in cols:
    if dev_df_conti[i].nunique()>10: ### taking out nunique values of conti cols which is greater than 10
        dev_df_conti[i]=pd.qcut(dev_df_conti[i],100,duplicates="drop")## converting conti into percentile value


# In[99]:


# An utility function to create dummy variable
data_dummi  = pd.get_dummies(dev_df_cat, prefix = 'colnames', drop_first = True)


# In[100]:


data_dummi.head()


# In[101]:


data_comb=pd.concat([data_dummi,dev_df_conti],axis=1)
data_comb


# In[96]:


data_comb.info()


# In[102]:


cols1 = list(data_comb.columns)
cols1.remove("cc_cons")


# In[103]:


#running a loop in concat df except id and cc_cons doing the target encoding 
for i in cols1:
    df1=pd.pivot_table(data_comb,index=i,values="cc_cons",aggfunc=np.median)
    df1.reset_index(inplace=True)
    df1.columns=[i, i+"_encoded"]
    data_comb= pd.merge(data_comb,df1,on=i).drop(i,axis=1)


# In[104]:


data_comb


# In[105]:


##Separating actual and predicted values in target variables.
data_comb0= data_comb[data_comb["cc_cons"].isna()==False]
test= data_comb[data_comb["cc_cons"].isna()]


# In[106]:


data_comb0["cc_cons"] = np.log(data_comb0["cc_cons"]+1) #becz value is getting bigger numbers we applied log to get smaller numbers.
#new_df_1["cc_cons"] = (new_df_1["cc_cons"]-np.mean(new_df_1["cc_cons"]))/np.var(new_df_1["cc_cons"])**0.5


# In[108]:


#taking out vif if theres any...
data_combdf = data_comb0[data_comb.columns.difference(["cc_cons"])]
vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data_combdf.values,i) for i in range(data_combdf.shape[1])]
vif["features"] = data_combdf.columns


# In[109]:


vif


# In[110]:


y=data_comb0[['cc_cons']] #target value
x= data_combdf


# In[111]:


from sklearn.model_selection import train_test_split


# In[208]:


#Splitting the data for sklearn methods
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=123)


# In[212]:


#23 important vars..
imp_var


# In[213]:


#23 important vars..
train_x[imp_var].shape


# In[211]:


grid = {'max_depth':np.arange(3,6),
       'max_features':np.arange(1,120),
       'criterion':['mse','friedman_mse']}


# In[147]:


tree = GridSearchCV(DecisionTreeRegressor(), grid, cv = 5,scoring ="neg_root_mean_squared_error")
tree.fit( train_x[imp_var], train_y )  


# In[148]:


tree.best_params_


# In[152]:


model = DecisionTreeRegressor(max_depth=3,criterion="friedman_mse",max_features=2)


# In[153]:


model.fit(train_x[imp_var],train_y)


# In[155]:


tr = np.exp(metrics.mean_squared_error(train_y,model.predict(train_x[imp_var]),squared = False))


# In[157]:


tre = np.exp(metrics.mean_squared_error(test_y,model.predict(test_x[imp_var]),squared = False))


# In[161]:


print("Train_RMSE:" ,tr,'Test_RMSE:',tre)


# ##### we tried to solve our problem with help of DecisionTreeRegressor model which is performing good. where, we achiveing Train_RMSE: 4.611 and Test_RMSE : 4.689, so we tried others model as well to get best RMSE value.
# 

# #### LinearRegression

# In[167]:


#LinearRegression

lr = LinearRegression()
lr.fit(train_x[imp_var],train_y)


# In[170]:


L = np.mean(metrics.mean_squared_error(train_y,lr.predict(train_x[imp_var]),squared=False))


# In[171]:


Le =np.mean(metrics.mean_squared_error(test_y,lr.predict(test_x[imp_var]),squared=False))


# In[185]:


print('Train_RMSE:',L,'Test_RMSE:',Le)


# ##### Here,we tried to solve our problem with help of Linear Regression using and here we actually getting perfect model even better then DecisionTreeRegressor. where, we achiveing Train_RMSE: 1.31 and Test_RMSE : 1.30.

# ##### predicted data for next 3 months

# In[174]:


predictions = lr.predict(predict_0[imp_var])


# In[175]:


test['cc_cons']=predictions


# In[177]:


predictions.shape


# In[183]:


comb_data = pd.concat([data_comb,test])


# In[184]:


comb_data['cc_cons']


# In[ ]:




