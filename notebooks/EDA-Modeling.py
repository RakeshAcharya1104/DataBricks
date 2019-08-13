# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

## to check the files in DBFS Path
display(dbutils.fs.ls('/FileStore/tables/'))

# COMMAND ----------

##Read the data from the DBFS system into spark
sp_bk_train = spark.read.format('csv').options(header='true', inferSchema='false').load('dbfs:/FileStore/tables/BikeSharing_train-431b4.csv')

# COMMAND ----------

##shape of the data
print("No of Rows {} and columns {}".format(sp_bk_train.count(),len(sp_bk_train.columns)))


# COMMAND ----------

##column names
print(sp_bk_train.columns)

# COMMAND ----------

##Reading the first 5 rows
sp_bk_train.head(n=5)

# COMMAND ----------

##converting to pandas dataframe
pd_bk_train = sp_bk_train.toPandas()

# COMMAND ----------

## datatypes of columns
pd_bk_train.dtypes

# COMMAND ----------

# datetime - hourly date + timestamp  
# season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
# holiday - whether the day is considered a holiday
# workingday - whether the day is neither a weekend nor holiday
# weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
#           2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
#           3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
#           4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# temp - temperature in Celsius
# atemp - "feels like" temperature in Celsius
# humidity - relative humidity
# windspeed - wind speed
# casual - number of non-registered user rentals initiated
# registered - number of registered user rentals initiated
# count - number of total rentalsy - whether the day is neither a weekend nor holiday

# COMMAND ----------

###converting the datatypes
pd_bk_train[['temp','atemp','humidity','windspeed','casual','registered','count']] = pd_bk_train[['temp','atemp','humidity','windspeed','casual','registered','count']].apply(pd.to_numeric)


# COMMAND ----------

## checking the distribution of data
pd_bk_train.describe()

# COMMAND ----------

## mapping the columns "Season" and "Weather"
pd_bk_train['season'] = pd_bk_train['season'].map({'1': "spring", '2' : "summer", '3' : "fall", '4' : "winter"})
pd_bk_train['weather'] = pd_bk_train['weather'].map({'1':"Clear", '2': "Mist Cloudy", '3': "Light Rain", '4': "Heavy Rain"})

# COMMAND ----------

from datetime import datetime
import calendar

# COMMAND ----------

###creating the new columns from datetime like month_name, year , week day name and hour
print(pd_bk_train['datetime'][2070])
print(datetime.strptime(pd_bk_train['datetime'][2070],'%Y-%m-%d %H:%M:%S').year)
calendar.day_name[datetime.strptime(pd_bk_train['datetime'][2070],'%Y-%m-%d %H:%M:%S').weekday()]

# COMMAND ----------

###creating the new columns from datetime like month_name, year , week day name and hour
pd_bk_train['MonthName'] = pd_bk_train['datetime'].apply(lambda x : calendar.month_name[datetime.strptime(x,'%Y-%m-%d %H:%M:%S').month])
pd_bk_train['Year'] = pd_bk_train['datetime'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d %H:%M:%S').year)
pd_bk_train['WeekDay'] = pd_bk_train['datetime'].apply(lambda x : calendar.day_name[datetime.strptime(x,'%Y-%m-%d %H:%M:%S').weekday()])
pd_bk_train['Hour'] = pd_bk_train['datetime'].apply(lambda x: x.split(" ")[1].split(":")[0])


# COMMAND ----------

pd_bk_train.head(10)

# COMMAND ----------

##removing the datetime column
pd_bk_train.drop(labels = ['datetime','casual','registered'],axis=1,inplace = True)

# COMMAND ----------

### checking the data types of new columns
pd_bk_train.dtypes

# COMMAND ----------

##converting year to factor or object
pd_bk_train['Year'] = pd_bk_train['Year'].astype('category')

# COMMAND ----------

### checking the NAN Values
pd_bk_train.isnull().sum()

# COMMAND ----------

### Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# COMMAND ----------

### Monthly and Weekday distribution of bike rents
plt.figure(1,figsize = (12,4))
sns.barplot(data=pd_bk_train,x="MonthName",y="count",estimator = sum)
display(plt.show())

# COMMAND ----------

### Weekday distribution of bike rents
plt.figure(1,figsize = (12,4))
sns.barplot(data=pd_bk_train,x="WeekDay",y="count",estimator = sum)
plt.ylabel("No of bike rentals")
display(plt.show())

# COMMAND ----------

##checking the distribution of Bike rents count
fig,axes = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(12, 10)
sns.distplot(pd_bk_train["count"],ax=axes[0][0])
stats.probplot(pd_bk_train["count"], dist='norm', fit=True, plot=axes[0][1])
sns.distplot(np.log(pd_bk_train["count"]),ax=axes[1][0])
stats.probplot(np.log1p(pd_bk_train["count"]), dist='norm', fit=True, plot=axes[1][1])
display(plt.show())

# COMMAND ----------

fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=pd_bk_train,y="count",orient="v",ax=axes[0][0])
sns.boxplot(data=pd_bk_train,y="count",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=pd_bk_train,y="count",x="Hour",orient="v",ax=axes[1][0])
sns.boxplot(data=pd_bk_train,y="count",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")

display(plt.show())

# COMMAND ----------

fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
fig.set_size_inches(10, 5)
sns.regplot(x="temp", y="count", data=pd_bk_train,ax=ax1)
sns.regplot(x="windspeed", y="count", data=pd_bk_train,ax=ax2)
sns.regplot(x="humidity", y="count", data=pd_bk_train,ax=ax3)
display(plt.show())

# COMMAND ----------

##correlation on numeric values
corrMatt = pd_bk_train[["temp","atemp","humidity","windspeed","count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(10,5)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
display(plt.show())


# COMMAND ----------

###removing the atemp column
pd_bk_train.drop('atemp',axis=1,inplace=True)

# COMMAND ----------

pd_bk_train_onehot = pd_bk_train.copy()
pd_bk_train_onehot = pd.get_dummies(pd_bk_train_onehot, columns=['season', 'weather','MonthName','WeekDay', 'Hour'])


# COMMAND ----------

### preparing X and Y dataframes
X = pd_bk_train_onehot.drop('count',axis =1)
Y = pd_bk_train_onehot['count']

# COMMAND ----------

from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# COMMAND ----------

##splitting the data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)

# COMMAND ----------

regr = RandomForestRegressor(n_estimators=50,max_depth=5, random_state=100,)
regr.fit(X_train, Y_train)

# COMMAND ----------

preds = regr.predict(X_train)

# COMMAND ----------

regr.score(X_train,Y_train)

# COMMAND ----------

