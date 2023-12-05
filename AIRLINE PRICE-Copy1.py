#!/usr/bin/env python
# coding: utf-8

# In[58]:


import os


# In[59]:


os.getcwd()


# In[60]:


os.chdir("D:\PYTHON\Practice")


# In[61]:


os.getcwd()


# In[62]:


import pandas as pd 
import numpy as np


# In[63]:


# QUESTION : 1
# PREDICT THE PRICE OF AIRLINE SURVICE ? 


# In[64]:


Airline_data=pd.read_csv("C:/Users/AYAN/Downloads/DATASETS/Hackathontrain.csv",encoding = 'Latin-1')
Airline_data.head(5)


# In[65]:


Airline_data.nunique()


# In[66]:


Airline_data.shape


# In[67]:


column = ["Distance","Aircraft_Type","Number_of_Stops","Day_of_Week","Month_of_Travel","Holiday_Season","Demand",
"Passenger_Count","Promotion_Type","Fuel_Price","Flight_Price"]
Airline_Data=Airline_data[column]
Airline_Data.head(5)


# In[68]:


Airline_Data.nunique()


# In[69]:


Catagory_clm=["Aircraft_Type","Number_of_Stops","Day_of_Week","Month_of_Travel","Holiday_Season","Demand","Promotion_Type"]
Conti_clm=["Distance","Passenger_Count","Fuel_Price","Flight_Price"]


# In[70]:


Airline_Data.hist(Conti_clm,figsize=(20,5))


# In[71]:


import matplotlib.pyplot as plt


# In[72]:


def dist_check(data, colTOplot):
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    import matplotlib.pyplot as plt
    
    fig, position=plt.subplots(nrows = 1 ,ncols=len(colTOplot),figsize=(27,5))
    plt.suptitle('Distribution of variables in'+ str(colTOplot))
    
    for colname , i in zip(colTOplot,range(len(colTOplot))):
        data.groupby(colname).size().plot(kind='bar',ax = position[i])
    


# In[73]:


dist_check(Airline_Data,Catagory_clm)


# In[74]:


Airline_Data.isnull().sum()


# In[75]:


import warnings
warnings.filterwarnings('ignore')


# In[76]:


print(Airline_Data.shape)
Airline_Data=Airline_Data.drop_duplicates()
print(Airline_Data.shape)


# In[ ]:


Airline_Data.to_csv("D:/PYTHON/Practice/Airline_data.csv")


# In[78]:


Airline_data.head(5)


# In[79]:


AirLine_Data=pd.read_csv("D:\PYTHON\Practice\Airline_data.csv",encoding='Latin-1')
AirLine_Data.head(5)


# In[80]:


AirLine_Data.isnull().sum()


# In[81]:


AirLine_Data['Fuel_Price'].fillna(value=AirLine_Data['Fuel_Price'].median(),inplace=True)


# In[82]:


AirLine_Data['Fuel_Price'].median()


# In[83]:


# next step find correlation with target variable


# In[84]:


import matplotlib.pyplot as plt


# In[85]:


Conti_clm=["Distance","Passenger_Count","Fuel_Price","Flight_Price"]
import matplotlib.pyplot as plt

for i in Conti_clm:
    AirLine_Data.plot.scatter(x=i,y='Flight_Price',figsize=(10,5),title= i +"vs"+"Flight_Price")


# In[86]:


Conti_clm=["Distance","Passenger_Count","Fuel_Price","Flight_Price"]

Corr_Data=AirLine_Data[Conti_clm].corr()
Corr_Data


# In[87]:


Corr_Data['Flight_Price'][abs(Corr_Data['Flight_Price'])>0.5]


# In[88]:


abs(Corr_Data['Flight_Price'])>0.5


# In[89]:


Catagory_clm=["Aircraft_Type","Number_of_Stops","Day_of_Week","Month_of_Travel","Holiday_Season","Demand","Promotion_Type"]
fig, plotcanvas=plt.subplots(nrows=len(Catagory_clm),ncols=1,figsize=(10,50))

for colname , i in zip(Catagory_clm,range(len(Catagory_clm))):
    AirLine_Data.boxplot(column="Flight_Price",by=colname, vert = True ,  ax = plotcanvas[i])


# In[90]:


def  Annova_test(Data,Target_Variable,Catagory_clm):
    from scipy.stats import f_oneway
    selected_column=[]
    
    
    for predictors in Catagory_clm:
        grouped_Data=Data.groupby(predictors)[Target_Variable].apply(list)
        Annova_result=f_oneway(*grouped_Data)
        
        if (Annova_result[1] < 0.05):
            print(predictors,"is correlated with",Target_Variable,"p| value:",
                 Annova_result)
            selected_column.append(predictors)
        else:
            print(predictors,"is not correlated with " +Target_Variable,"p| value:",
                 Annova_result)
    return(selected_column)
    
        


# In[91]:


Catagory_clm=["Aircraft_Type","Number_of_Stops","Day_of_Week","Month_of_Travel","Holiday_Season","Demand","Promotion_Type"]
Annova_test(Data=AirLine_Data,
           Target_Variable="Flight_Price",
           Catagory_clm=Catagory_clm)


# In[137]:


final_clm=['Aircraft_Type','Number_of_Stops','Day_of_Week','Month_of_Travel','Holiday_Season','Demand','Distance']
Final_Data=AirLine_Data[final_clm]
Final_Data.head(5)


# In[93]:


Final_Data.nunique()


# In[138]:


Final_Data['Day_of_Week'].replace({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7},
                                   inplace=True)



# In[139]:


Final_Data.head(5)


# In[140]:


Final_Data['Aircraft_Type'].replace({'Airbus A320':1,'Airbus A380':2,'Boeing 737':3,'Boeing 777':4,'Boeing 787':5},
                                   inplace=True)

Final_Data['Holiday_Season'].replace({'Fall':1,'None':2,'Spring':3,'Summer':4,'Winter':5},
                                   inplace=True)


Final_Data['Demand'].replace({'Low':0,'Medium':1,'High':2},inplace=True)


# In[141]:


Final_Data.nunique()


# In[142]:


Final_Data.head(5)


# In[143]:


Final_Data['Month_of_Travel']=Final_Data['Month_of_Travel'].str.strip()
Final_Data['Month_of_Travel'].replace({'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8
                                       ,'September':9,'October':10,'November':11,'December':12},
                                      inplace=True)


# In[144]:


Final_Data.head(5)
Final_Data['Flight_Price']=AirLine_Data['Flight_Price']
Final_Data.head(5)


# In[145]:


from sklearn.model_selection import train_test_split


# In[146]:


Target = 'Flight_Price'
final_clm = ['Aircraft_Type', 'Number_of_Stops', 'Day_of_Week', 'Month_of_Travel', 'Holiday_Season', 'Demand', 'Distance']

X = Final_Data[final_clm].values  # X contains the features
y = Final_Data[Target].values  # y contains the target variable

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[147]:


X_train[0:10]


# In[148]:


y_train[0:10]


# In[149]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


# In[166]:


Regmodel=LinearRegression()
RegModel=tree.DecisionTreeRegressor(max_depth=3)

DT=RegModel.fit(X_train,y_train)
LRG=Regmodel.fit(X_train,y_train)
Prediction=LRG.predict(X_test)


# Create a DataFrame for predictions
Prediction_result = pd.DataFrame(data=X_test, columns=final_clm)
Prediction_result[Target] = y_test
Prediction_result['Predicted_' + Target] = np.round(Prediction)
print(Prediction_result.head(10))

# Modify the custom scoring function
def Accuracy_score(Target, prediction):
    MAPE = (100 * (abs(Target - prediction) / Target))
    return 100 - np.mean(MAPE)

# Use the modified custom scoring function in cross_val_score
Custom_Scoring = make_scorer(Accuracy_score, greater_is_better=True)
Prominent_Accuracy = cross_val_score(Regmodel, X, y, cv=10, scoring=Custom_Scoring)

print('\nAccuracy values of 10 fold cross-validation:\n', Prominent_Accuracy)
print('\nMean Accuracy of the model:\n', round(Prominent_Accuracy.mean(), 2))

get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(DT.feature_importances_, index=final_clm)
feature_importances.nlargest(10).plot(kind='barh')


# In[167]:


Prediction_result.head(10)


# In[168]:


LRG.coef_


# In[169]:


LRG.intercept_


# In[170]:


#Flight_price=1.81*(Aircraft_Type)+4.59*(Number_of_stops)+6.24*(Day_of_Week)-6.46*(Month_of_Travel)+4.03*(Holiday_Season)+
#9.38*(Demand)+4.49*(Distance) - 227.43


# In[174]:


Prediction_result.to_csv("D:/PYTHON/Practice/Prediction_result.csv",index=False)


# In[ ]:




