import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import datetime

data=pd.read_excel(r'C:/...../Data_Train.xlsx')
data.head()
data.shape
# if using in jupyter
pd.set_option('display.max_columns',None)
# Deal with missing values
# Data cleaning for data analysis and modelling

data.isna().sum() 
data.dropna(inplace=True) # bydefault its axis=0

data.dtypes

def change_to_datetime(col):
    data[col]=pd.to_datetime(data[col])

data.columns
cols=['Date_of_Journey','Dep_Time','Arrival_Time']
for col in cols:
    change_to_datetime(col)
 
#1 Date_of_Journey 
data['Journey_day']=data['Date_of_Journey'].dt.day
data['Journey_month']=data['Date_of_Journey'].dt.month
data.drop('Date_of_Journey', axis=1, inplace=True)


def extract_hour(col):
    data[col+'_hour']=data[col].dt.hour
    
def extract_minute(col):
    data[col+'_minute']=data[col].dt.minute

def drop_col(df,col):
    df.drop(col, axis=1, inplace=True)
    
#2 Dep_Time
extract_hour('Dep_Time')
extract_minute('Dep_Time')
drop_col(data,'Dep_Time')
#3 Arrival_Time
extract_hour('Arrival_Time')
extract_minute('Arrival_Time')
drop_col(data,'Arrival_Time')

duration=list(data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split())==1:
        if 'h' in duration[i]:
            duration[i] = duration[i]+' 0m'
        else:
            print(duration[i])
            print(i)
            duration[i] = '0h '+duration[i]
    else:
        continue

data['Duration']= duration


#duration[0][0:-1]
def hour(x):
    return x.split(' ')[0][0:-1]

def minute(x):
    return x.split(' ')[1][0:-1]


data['Duration_hours']= data['Duration'].apply(hour)
data['Duration_minute']= data['Duration'].apply(minute)
drop_col(data,'Duration')

data.dtypes
data['Duration_hours'] = data['Duration_hours'].astype('int')
data['Duration_minute'] = data['Duration_minute'].astype('int')

cat= [col for col in data.columns if data[col].dtype=='O']
cat
ncat = [col for col in data.columns if data[col].dtype!='O']
ncat
#Nominal data = categorical data  : use one_hot_encoding
#Ordinal data = non-cattegorical/ hierarchial data  : use label_encoder
categorical= data[cat]
categorical
'''
def convert_to_category(col):
    global column
    column = pd.get_dummies(data[col], drop_first=True)
    return column
'''

categorical['Airline'].value_counts()  
plt.figure(figsize=(15,5))
sb.boxplot(x='Airline',y='Price', data=data.sort_values('Price', ascending=False))
sb.catplot(x='Airline',y='Price', data=data.sort_values('Price', ascending=False), kind ='boxen', height=6, aspect=3)
Airline = pd.get_dummies(data['Airline'], drop_first=True)

categorical['Source'].value_counts()
plt.figure(figsize=(15,5))
sb.boxplot(x='Source',y='Price', data=data.sort_values('Price', ascending=False))
sb.catplot(x='Source',y='Price', data=data.sort_values('Price', ascending=False), kind ='boxen', height=6, aspect=3)
Source = pd.get_dummies(data['Source'], drop_first=True)

categorical['Destination'].value_counts()
plt.figure(figsize=(15,5))
sb.boxplot(x='Destination',y='Price', data=data.sort_values('Price', ascending=False))
sb.catplot(x='Destination',y='Price', data=data.sort_values('Price', ascending=False), kind ='boxen', height=6, aspect=3)
Destination = pd.get_dummies(data['Destination'], drop_first=True)

categorical['Total_Stops'].value_counts()
plt.figure(figsize=(15,5))
sb.boxplot(x='Total_Stops',y='Price', data=data.sort_values('Price', ascending=False))
sb.catplot(x='Total_Stops',y='Price', data=data.sort_values('Price', ascending=False), kind ='boxen', height=6, aspect=3)
Total_Stops = pd.get_dummies(data['Total_Stops'], drop_first=True)

categorical['Route_1'] = categorical['Route'].str.split('→').str[0]
categorical['Route_2'] = categorical['Route'].str.split('→').str[1]
categorical['Route_3'] = categorical['Route'].str.split('→').str[2]
categorical['Route_4'] = categorical['Route'].str.split('→').str[3]
categorical['Route_5'] = categorical['Route'].str.split('→').str[4]

drop_col(categorical,'Route')

categorical.isna().sum()
categorical.columns
cat_col=['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5']

for i in cat_col:
    categorical[i].fillna('None',inplace=True)


for i in categorical:
    print('{} has {} categories'.format(i, len(categorical[i].value_counts())))

#categorical['Additional_Info'].value_counts()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

for i in cat_col:
    categorical[i] = le.fit_transform(categorical[i])

drop_col(categorical,'Additional_Info')

categorical['Total_Stops'].value_counts()

categorical.replace({'non-stop': 0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4},inplace=True)
#or
#dict= {'non-stop': 0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}
#categorical['Total_Stops'] =categorical['Total_Stops'].map(dict)

data = pd.concat([categorical,Airline,Source,Destination,data[ncat]],axis=1)

data.head()
data.columns
drop_col(data,'Airline')
drop_col(data,'Source')
drop_col(data,'Destination')

a=data.corr()
plt.figure(figsize=(18,18))
sb.heatmap(a, annot=True, cmap='RdYlGn')
plt.show()

def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sb.distplot(df[col], ax=ax1)
    sb.boxplot(df[col], ax=ax2)

plot(data,'Price')

data['Price']=np.where(data['Price']>40000, data['Price'].median(),data['Price'])

ind= data.drop('Price', axis=1)
dep= data['Price']


from sklearn.feature_selection import mutual_info_classif
#mutual_info_classif(ind,dep)
imp=pd.DataFrame(mutual_info_classif(ind,dep), index=ind.columns)
imp.columns=['importance']
imp.sort_values(by='importance', ascending=False)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(ind,dep,test_size=0.2)

from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
def predict(my_model):
    model=my_model.fit(xtrain,ytrain)
    print('Training Score are : {}'.format(model.score(xtrain,ytrain)))
    ypred= model.predict(xtest)
    print('Predictions are : {}'.format(ypred))
    print('\n')
    #print('Accuracy_score is : ',accuracy_score(ytest, ypred))
    r2_score= metrics.r2_score(ytest,ypred)
    print('r2_score : {}'.format(r2_score))
    print('MAE : ',mean_absolute_error(ytest, ypred))
    print('MSE : ',mean_squared_error(ytest, ypred))
    print('RMSE : ',np.sqrt(mean_squared_error(ytest, ypred)))
    sb.distplot(ytest-ypred)
    #sb.distplot(ytest)
    #sb.distplot(ypred)

from sklearn.ensemble import RandomForestRegressor
predict(RandomForestRegressor())

from sklearn.tree import DecisionTreeRegressor
predict(DecisionTreeRegressor())
'''
#waste dont use >0.4,0.6,0.3
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
predict(BernoulliNB())
predict(GaussianNB())
predict(MultinomialNB())
'''





















