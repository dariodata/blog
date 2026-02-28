---
title: "Fitbit activity and sleep data: a time-series analysis with Generalized Additive Models"
date: 2018-04-01
description: "This is a time-series analysis of activity and sleep data from a fitbit user throughout a year. I use this data to predict an additional year of the life of the user using Generalized Additive Models."
tags: [time series, wearables, forecasting]
math: true
---

The goal of this notebook is to provide an analysis of the time-series data from a user of a fitbit tracker throughout a year. I will use this data to predict an additional year of the life of the user using [Generalized Additive Models](https://en.wikipedia.org/wiki/Generalized_additive_model).

[Data source](https://algo-data.quora.com/Data-sets-of-any-type-some-links): [Activity](https://drive.google.com/open?id=0Bx4yoK5aogTSbGJ2WlkwYjlHejQ), [Sleep](https://drive.google.com/open?id=0Bx4yoK5aogTSMUFqRjVNcko5WlU)

Packages used:
- pandas, numpy, matplotlib, seaborn
- [Prophet](https://github.com/facebook/prophet)


```python
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

## Data cleaning (missing data and outliers)


```python
# import the activity data
activity = pd.read_csv('OneYearFitBitData.csv')
# change commas to dots
activity.iloc[:,1:] = activity.iloc[:,1:].applymap(lambda x: float(str(x).replace(',','.')))
# change column names to English
activity.columns = ['Date', 'BurnedCalories', 'Steps', 'Distance', 'Floors', 'SedentaryMinutes', 'LightMinutes', 'ModerateMinutes', 'IntenseMinutes', 'IntenseActivityCalories']
# import the sleep data
sleep = pd.read_csv('OneYearFitBitDataSleep.csv')
# check the size of the dataframes
activity.shape, sleep.shape
# merge dataframes
data = pd.merge(activity, sleep, how='outer', on='Date')
# parse date into correct format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
# correct units for Calories and Steps
for c in ['BurnedCalories', 'Steps', 'IntenseActivityCalories']:
    data[c] = data[c]*1000
```

Once imported, we should check for any missing data:


```python
# check for missing data
data.isnull().sum()
```




    Date                       0
    BurnedCalories             0
    Steps                      0
    Distance                   0
    Floors                     0
    SedentaryMinutes           0
    LightMinutes               0
    ModerateMinutes            0
    IntenseMinutes             0
    IntenseActivityCalories    0
    MinutesOfSleep             5
    MinutesOfBeingAwake        5
    NumberOfAwakings           5
    LengthOfRestInMinutes      5
    dtype: int64




```python
# check complete rows where sleep data is missing
data.iloc[np.where(data['MinutesOfSleep'].isnull())[0],:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BurnedCalories</th>
      <th>Steps</th>
      <th>Distance</th>
      <th>Floors</th>
      <th>SedentaryMinutes</th>
      <th>LightMinutes</th>
      <th>ModerateMinutes</th>
      <th>IntenseMinutes</th>
      <th>IntenseActivityCalories</th>
      <th>MinutesOfSleep</th>
      <th>MinutesOfBeingAwake</th>
      <th>NumberOfAwakings</th>
      <th>LengthOfRestInMinutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-05-08</td>
      <td>1934.0</td>
      <td>905000.0</td>
      <td>0.65</td>
      <td>0.0</td>
      <td>1.355</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>168000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>275</th>
      <td>2016-02-01</td>
      <td>2986.0</td>
      <td>11426.0</td>
      <td>8.52</td>
      <td>12.0</td>
      <td>911.000</td>
      <td>192.0</td>
      <td>48.0</td>
      <td>43.0</td>
      <td>1478.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>276</th>
      <td>2016-02-02</td>
      <td>2974.0</td>
      <td>10466.0</td>
      <td>7.78</td>
      <td>13.0</td>
      <td>802.000</td>
      <td>152.0</td>
      <td>48.0</td>
      <td>48.0</td>
      <td>1333.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>277</th>
      <td>2016-02-03</td>
      <td>3199.0</td>
      <td>12866.0</td>
      <td>9.63</td>
      <td>11.0</td>
      <td>767.000</td>
      <td>271.0</td>
      <td>45.0</td>
      <td>28.0</td>
      <td>1703.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>278</th>
      <td>2016-02-04</td>
      <td>2037.0</td>
      <td>2449.0</td>
      <td>1.87</td>
      <td>0.0</td>
      <td>821.000</td>
      <td>80.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>337000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the sleep information was missing for some dates. The activity information for those days is complete. Therefore, we should not get rid of those rows just now.


```python
# check rows for which steps count is zero
data.iloc[np.where(data['Steps']==0)[0],:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BurnedCalories</th>
      <th>Steps</th>
      <th>Distance</th>
      <th>Floors</th>
      <th>SedentaryMinutes</th>
      <th>LightMinutes</th>
      <th>ModerateMinutes</th>
      <th>IntenseMinutes</th>
      <th>IntenseActivityCalories</th>
      <th>MinutesOfSleep</th>
      <th>MinutesOfBeingAwake</th>
      <th>NumberOfAwakings</th>
      <th>LengthOfRestInMinutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>235</th>
      <td>2015-12-23</td>
      <td>1789.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.44</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>317</th>
      <td>2016-03-13</td>
      <td>1790.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.44</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>332</th>
      <td>2016-03-28</td>
      <td>1790.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.44</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



We can also see that the step count for some datapoints is zero. If we look at the complete rows, we can see that on those days nearly no other data was recorded. I assume that the user probably did not wear the fitness tracker on that day and we could get rid of those complete rows.


```python
# drop days with a step count of zero
data = data.drop(np.where(data['Steps']==0)[0], axis=0)
```


```python
# plot the distribution of data for step count
sns.distplot(data['Steps'])
plt.title('Histogram for step count')
```




    <matplotlib.text.Text at 0x10fb34cc0>




    
![png](/images/2018-04-01-fitbit_prophet_files/2018-04-01-fitbit_prophet_12_1.png)
    


Step count is probably the most accurate measure obtained from a pedometer. Looking at the distribution of this variable, however, we can see that there is a chance that we have outliers in the data, as at least one value seems to be much higher than all the rest.


```python
# sort data by step count in a descending order
data.sort_values(by='Steps', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BurnedCalories</th>
      <th>Steps</th>
      <th>Distance</th>
      <th>Floors</th>
      <th>SedentaryMinutes</th>
      <th>LightMinutes</th>
      <th>ModerateMinutes</th>
      <th>IntenseMinutes</th>
      <th>IntenseActivityCalories</th>
      <th>MinutesOfSleep</th>
      <th>MinutesOfBeingAwake</th>
      <th>NumberOfAwakings</th>
      <th>LengthOfRestInMinutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-05-08</td>
      <td>1934.0</td>
      <td>905000.0</td>
      <td>0.65</td>
      <td>0.0</td>
      <td>1.355</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>168000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>267</th>
      <td>2016-01-24</td>
      <td>1801.0</td>
      <td>39000.0</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>1.076</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16000.0</td>
      <td>342.0</td>
      <td>48.0</td>
      <td>31.0</td>
      <td>390.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2015-06-13</td>
      <td>4083.0</td>
      <td>26444.0</td>
      <td>19.65</td>
      <td>22.0</td>
      <td>549.000</td>
      <td>429.0</td>
      <td>56.0</td>
      <td>56.0</td>
      <td>2818.0</td>
      <td>169.0</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>196.0</td>
    </tr>
    <tr>
      <th>363</th>
      <td>2016-04-28</td>
      <td>4030.0</td>
      <td>25571.0</td>
      <td>19.30</td>
      <td>15.0</td>
      <td>606.000</td>
      <td>293.0</td>
      <td>42.0</td>
      <td>129.0</td>
      <td>2711.0</td>
      <td>374.0</td>
      <td>56.0</td>
      <td>34.0</td>
      <td>430.0</td>
    </tr>
    <tr>
      <th>320</th>
      <td>2016-03-16</td>
      <td>3960.0</td>
      <td>25385.0</td>
      <td>20.45</td>
      <td>17.0</td>
      <td>638.000</td>
      <td>254.0</td>
      <td>17.0</td>
      <td>124.0</td>
      <td>2556.0</td>
      <td>368.0</td>
      <td>46.0</td>
      <td>22.0</td>
      <td>414.0</td>
    </tr>
  </tbody>
</table>
</div>



We found the outlier! It seems that the step count for the first day (our data starts on May 8th, 2015) is too high to be a correct value for the amount of steps taken by the user on that day. Maybe the device saves the vibration since its production as step count which is loaded on the first day that the user wears the tracker. We can anyway get rid of that row since the sleep data is also not available for this day.


```python
# drop outlier
data = data.drop(np.where(data['Steps']>=100000)[0], axis=0)
```

Now we can look at our preprocessed data. Shape, distribution of the variables, and a look at some rows from the dataframe, are all useful things to observe:


```python
data.shape
```




    (369, 14)




```python
fig, ax = plt.subplots(5,3, figsize=(8,10))

for c, a in zip(data.columns[1:], ax.flat):
    df = pd.DataFrame()
    df['ds'] = data['Date']
    df['y'] = data[c]
    df = df.dropna(axis=0, how='any')
    sns.distplot(df['y'], axlabel=False, ax=a)
    a.set_title(c)

plt.suptitle('Histograms of variables from fitbit data', y=1.02, fontsize=14);
plt.tight_layout()
```


    
![png](/images/2018-04-01-fitbit_prophet_files/2018-04-01-fitbit_prophet_19_0.png)
    



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>BurnedCalories</th>
      <th>Steps</th>
      <th>Distance</th>
      <th>Floors</th>
      <th>SedentaryMinutes</th>
      <th>LightMinutes</th>
      <th>ModerateMinutes</th>
      <th>IntenseMinutes</th>
      <th>IntenseActivityCalories</th>
      <th>MinutesOfSleep</th>
      <th>MinutesOfBeingAwake</th>
      <th>NumberOfAwakings</th>
      <th>LengthOfRestInMinutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2015-05-09</td>
      <td>3631.0</td>
      <td>18925.0</td>
      <td>14.11</td>
      <td>4.0</td>
      <td>611.0</td>
      <td>316.0</td>
      <td>61.0</td>
      <td>60.0</td>
      <td>2248.0</td>
      <td>384.0</td>
      <td>26.0</td>
      <td>23.0</td>
      <td>417.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-05-10</td>
      <td>3204.0</td>
      <td>14228.0</td>
      <td>10.57</td>
      <td>1.0</td>
      <td>602.0</td>
      <td>226.0</td>
      <td>14.0</td>
      <td>77.0</td>
      <td>1719.0</td>
      <td>454.0</td>
      <td>35.0</td>
      <td>21.0</td>
      <td>491.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-05-11</td>
      <td>2673.0</td>
      <td>6756.0</td>
      <td>5.02</td>
      <td>8.0</td>
      <td>749.0</td>
      <td>190.0</td>
      <td>23.0</td>
      <td>4.0</td>
      <td>962000.0</td>
      <td>387.0</td>
      <td>46.0</td>
      <td>25.0</td>
      <td>436.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-05-12</td>
      <td>2495.0</td>
      <td>5020.0</td>
      <td>3.73</td>
      <td>1.0</td>
      <td>876.0</td>
      <td>171.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>736000.0</td>
      <td>311.0</td>
      <td>31.0</td>
      <td>21.0</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-05-13</td>
      <td>2760.0</td>
      <td>7790.0</td>
      <td>5.79</td>
      <td>15.0</td>
      <td>726.0</td>
      <td>172.0</td>
      <td>34.0</td>
      <td>18.0</td>
      <td>1094.0</td>
      <td>407.0</td>
      <td>65.0</td>
      <td>44.0</td>
      <td>491.0</td>
    </tr>
  </tbody>
</table>
</div>



## Predicting the step count for an additional year

In order to use the Prophet package to predict the future using a Generalized Additive Model, we need to create a dataframe with columns ```ds``` and ```y``` (we need to do this for each variable):
- ```ds``` is the date stamp data giving the time component
- ```y``` is the variable that we want to predict

In our case we will use the log transform of the step count in order to decrease the effect of outliers on the model.


```python
df = pd.DataFrame()
df['ds'] = data['Date']
df['y'] = data['Steps']
# log-transform of step count
df['y'] = np.log(df['y'])
```

Now we need to specify the type of growth model that we want to use:
- Linear: assumes that the variable ```y``` grows linearly in time (doesn't apply to our step count scenario, if the person sticks to their normal lifestyle)
- Logistic: assumes that the variable ```y``` grows logistically in time and saturates at some point

I will assume that the person, for whom we want to predict the step count in the following year, will not have any dramatic lifestyle changes that makes them start to walk more. Therefore, I am using logistic 'growth' capped to a cap of the mean of the data, which in practice means that the step count's growth trend will be 'zero growth'.


```python
df['cap'] = df['y'].median()
m = Prophet(growth='logistic', yearly_seasonality=True)
m.fit(df)
```

    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    /Users/dario/anaconda/envs/datasci/lib/python3.6/site-packages/pystan/misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(np.asarray(v).dtype, float):





    <fbprophet.forecaster.Prophet at 0x115274dd8>



After fitting the model, we need a new dataframe ```future``` with the additional rows for which we want to predict ```y```.


```python
future = m.make_future_dataframe(periods=365, freq='D')
future['cap'] = df['y'].median()
```

Now we can call predict on the fitted model and obtain relevant statistics for the forecast period. We can also plot the results.


```python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>729</th>
      <td>2017-05-03</td>
      <td>9.565727</td>
      <td>9.109546</td>
      <td>10.034228</td>
    </tr>
    <tr>
      <th>730</th>
      <td>2017-05-04</td>
      <td>9.547910</td>
      <td>9.089434</td>
      <td>10.038742</td>
    </tr>
    <tr>
      <th>731</th>
      <td>2017-05-05</td>
      <td>9.536875</td>
      <td>9.084160</td>
      <td>10.035884</td>
    </tr>
    <tr>
      <th>732</th>
      <td>2017-05-06</td>
      <td>9.525034</td>
      <td>9.068735</td>
      <td>10.008784</td>
    </tr>
    <tr>
      <th>733</th>
      <td>2017-05-07</td>
      <td>9.289238</td>
      <td>8.792235</td>
      <td>9.746812</td>
    </tr>
  </tbody>
</table>
</div>




```python
m.plot(forecast, ylabel='log(Steps)', xlabel='Date');
plt.title('1-year prediction of step count from 1 year of fitbit data');
```


    
![png](/images/2018-04-01-fitbit_prophet_files/2018-04-01-fitbit_prophet_30_0.png)
    


We can see that the model did a good job in mimicking the behavior of step count during the year for which the data was available. This seems reasonable, as we do not expect the pattern to vary necessarily, if the person continues to have a similar lifestyle.

Additionally, we can plot the components from the Generalized Additive Model and see their effect on the 'y' variable. In this case we have the general trend (remember we capped this at '10'), the yearly seasonality effect, and the weekly effect.


```python
m.plot_components(forecast);
plt.suptitle('GAM components for prediction of step count', y=1.02, fontsize=14);
```


    
![png](/images/2018-04-01-fitbit_prophet_files/2018-04-01-fitbit_prophet_33_0.png)
    


Here we see some interesting patterns:
- The general 'growth' trend is as expected, as we assumed that there would be no growth beyond the mean of the existing data.
- The yearly effect shows a trend towards higher activity during the summer months, however the variation is considerable, probably due to the fact that our dataset consisted of the data for one year only
- The weekly effect shows that Sunday is a day of lower activity for this person whereas Saturday is the day where the activity is the highest. So, grocery shopping on Saturday, Netflix on Sunday? :)

## Sleep analysis

A very important part of our lives is sleep. It would be very interesting to look at the sleep habits of the user of the fitness tracker and see if we can get some insights from this data.


```python
df = pd.DataFrame()
df['ds'] = data['Date']
df['y'] = data['MinutesOfSleep']
df = df.dropna(axis=0, how='any')
# drop rows where sleep time is zero, as this would mean that the person did not wear the tracker overnight and the data is missing
df = df.iloc[np.where(df['y']!=0)[0],:]
```


```python
# distribution of MinutesOfSleep
sns.distplot(df['y'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1163770f0>




    
![png](/images/2018-04-01-fitbit_prophet_files/2018-04-01-fitbit_prophet_38_1.png)
    



```python
df['cap'] = df['y'].median()
m = Prophet(growth='logistic', yearly_seasonality=True)
m.fit(df)

future = m.make_future_dataframe(periods=365, freq='D')
future['cap'] = df['y'].median()

forecast = m.predict(future)
m.plot(forecast);
plt.title('1-year prediction of MinutesOfSleep from 1 year of fitbit data');
```

    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    /Users/dario/anaconda/envs/datasci/lib/python3.6/site-packages/pystan/misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(np.asarray(v).dtype, float):



    
![png](/images/2018-04-01-fitbit_prophet_files/2018-04-01-fitbit_prophet_39_1.png)
    


The model again seems to predict a similar sleep behavior for the predicted year. This seems reasonable, as we do not expect the pattern to vary necessarily, if the person continues to have a similar lifestyle.


```python
m.plot_components(forecast);
plt.suptitle('GAM components for prediction of MinutesOfSleep', y=1.02, fontsize=14);
```


    
![png](/images/2018-04-01-fitbit_prophet_files/2018-04-01-fitbit_prophet_41_0.png)
    


A look at the amount of sleep reveals:
- A saturation trend at the median (we set this assumption)
- A yearly effect shows a trend towards higher amount of sleep during the summer months, with more variation during winter
- The weekly effect shows lowest sleep amount on Mondays (maybe going to bed late on Sunday and waking up early on Monday is a pattern for this user). Highest amout of sleep occurs on Saturdays (no alarm to wake up to on Saturday morning!). Interestingly, the user seems to get more sleep on Wednesdays than on Mondays or Tuesdays, which could mean that their work schedule is not constant during week-days.

### Appendix
As an exercise, I have plotted the predictions for the most interesing variables in the dataset. Enjoy!


```python
zeros_allowed = ['Floors', 'SedentaryMinutes', 'LightMinutes', 'ModerateMinutes', 'IntenseMinutes', 'IntenseActivityCalories', 'MinutesOfBeingAwake', 'NumberOfAwakings']

fig, ax = plt.subplots(3,3, figsize=(12,6), sharex=True)

predict_cols = ['Steps', 'Floors', 'BurnedCalories', 'LightMinutes', 'ModerateMinutes', 'IntenseMinutes', 'MinutesOfSleep', 'MinutesOfBeingAwake', 'NumberOfAwakings']

for c, a in zip(predict_cols, ax.flat):
    df = pd.DataFrame()
    df['ds'] = data['Date']
    df['y'] = data[c]
    df = df.dropna(axis=0, how='any')
    
    if c not in zeros_allowed:
        df = df.iloc[np.where(df['y']!=0)[0],:]

    df['cap'] = df['y'].median()
    m = Prophet(growth='logistic', yearly_seasonality=True)
    m.fit(df)

    future = m.make_future_dataframe(periods=365, freq='D')
    future['cap'] = df['y'].median()
    future.tail()

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    m.plot(forecast, xlabel='', ax=a);
    a.set_title(c)

    #m.plot_components(forecast);

plt.suptitle('1-year prediction per variable from 1 year of fitbit data', y=1.02, fontsize=14);
plt.tight_layout()
```

    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    /Users/dario/anaconda/envs/datasci/lib/python3.6/site-packages/pystan/misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(np.asarray(v).dtype, float):
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.



    
![png](/images/2018-04-01-fitbit_prophet_files/2018-04-01-fitbit_prophet_44_1.png)
    



```python

```
