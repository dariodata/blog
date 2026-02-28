# Visualizing parallel event series in Python
> In this post, I will use Python to visualize two different series of events, plotting them on top of each other to gain insights from time series data."

- toc:true
- branch: master
- badges: true
- comments: false
- author: Dario Arcos-Díaz
- categories: [time series, visualization]
- image: images/event-tracker_thumb.png

Do movie releases produce literal earthquakes? We always hear about new movie releases being a "blast", some sure are. But how do two independent events correlate with each other? In this post, I will use Python to visualize two different series of events, plotting them on top of each other to gain insights from time series data.


```python
# Imports
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Set2')
sns.set_style("whitegrid")
%matplotlib inline
```

# Getting the data

To make this example more fun, I decided to use two independent series of events for which data is readily available in the internet:

- [List of earthquakes around the world](http://world-earthquakes.com/index.php?option=eqs&year=2016)
- [List of film releases in the USA](http://www.firstshowing.net/schedule2016/)

## Clean and prepare earthquake data

We start by downloading the .csv export from the [world earthquake website](http://world-earthquakes.com/index.php?option=eqs&year=2016) to the 'data' directory and reading the file into a pandas DataFrame


```python
df = pd.read_csv('data/earthquakes_raw.csv', sep=';')
df.dropna((0,1), how='all', inplace=True)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Zone/Region</th>
      <th>Magnitude (Mw)</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>December 09</td>
      <td>SOLOMON ISLANDS</td>
      <td>6.9</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>December 08</td>
      <td>CALIFORNIA</td>
      <td>6.5</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>December 08</td>
      <td>SOLOMON ISLANDS</td>
      <td>7.8</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>December 06</td>
      <td>SUMATRA</td>
      <td>6.5</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>November 25</td>
      <td>CHINA</td>
      <td>6.6</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>



We have to unify the date information from the Date and Year columns. Then we can save the cleaned-up earthquake date data to a file 'data/earthquakes.csv'


```python
df['Date'] = df['Date'] + ' ' + df['Year'].map(str)
del df['Year']
df['Date'] = df['Date'].apply(lambda x: 
                              datetime.strptime(x, '%B %d %Y'))
df['Date'].to_csv('data/earthquakes.csv')
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Zone/Region</th>
      <th>Magnitude (Mw)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-12-09</td>
      <td>SOLOMON ISLANDS</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-12-08</td>
      <td>CALIFORNIA</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-12-08</td>
      <td>SOLOMON ISLANDS</td>
      <td>7.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-12-06</td>
      <td>SUMATRA</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-11-25</td>
      <td>CHINA</td>
      <td>6.6</td>
    </tr>
  </tbody>
</table>
</div>



## Clean and prepare movie release data

The movie release data was retrieved from [this website](http://www.firstshowing.net/schedule2016/) and saved to the 'data' directory. We then read the file into a pandas DataFrame. The resulting table tells us the release date and which movies where released on a that date (up to 5 movies).


```python
df = pd.read_csv('data/filmrelease_raw.csv', sep=';', header=None)
df.dropna((0,1), how='all', inplace=True, thresh=2)
df.columns = ['Date', 'Film1', 'Film2', 'Film3', 'Film4', 'Film5']
df.head(), df.tail()
```




    (                        Date              Film1            Film2  \
     0    Friday, January 9, 2015            Taken 3              NaN   
     4   Friday, January 16, 2015           Blackhat       Paddington   
     14  Friday, January 23, 2015          Mortdecai    Strange Magic   
     24  Friday, January 30, 2015     Black or White  Project Almanac   
     34  Friday, February 6, 2015  Jupiter Ascending      Seventh Son   
     
                                        Film3 Film4 Film5  
     0                                    NaN   NaN   NaN  
     4                     The Wedding Ringer   NaN   NaN  
     14                     The Boy Next Door   NaN   NaN  
     24                              The Loft   NaN   NaN  
     34  SpongeBob Movie: Sponge Out of Water   NaN   NaN  ,
                            Date                     Film1         Film2  \
     836      Friday, December 9  🎥 Office Christmas Party           NaN   
     840     Friday, December 16       🎥 Collateral Beauty  🎥 La La Land   
     850  Wednesday, December 21        🎥 Assassin's Creed  🎥 Passengers   
     863     Friday, December 23                🎥 Why Him?           NaN   
     867     Sunday, December 25                  🎥 Fences           NaN   
     
                                   Film3   Film4 Film5  
     836                             NaN     NaN   NaN  
     840  🎥 Rogue One: A Star Wars Story     NaN   NaN  
     850                  🎥 Patriots Day  🎥 Sing   NaN  
     863                             NaN     NaN   NaN  
     867                             NaN     NaN   NaN  )



Talk about raw _unclean_ data! It seems that, at the top of the table, the date information contains the year (2015). However, upon further inspection we can see that the bottom of the table does not show us the year anymore. From the website information we find out that, after the index 716 and onwards, the missing year information is '2016'. So we add this data to the DataFrame and change the date format to a more readable one.


```python
df.loc[lambda x: x.index >= 716, 'Date'] += ', 2016'
df['Date'] = df['Date'].apply(lambda x: 
                              datetime.strptime(x, '%A, %B %d, %Y'))
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Film1</th>
      <th>Film2</th>
      <th>Film3</th>
      <th>Film4</th>
      <th>Film5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-09</td>
      <td>Taken 3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-16</td>
      <td>Blackhat</td>
      <td>Paddington</td>
      <td>The Wedding Ringer</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-01-23</td>
      <td>Mortdecai</td>
      <td>Strange Magic</td>
      <td>The Boy Next Door</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015-01-30</td>
      <td>Black or White</td>
      <td>Project Almanac</td>
      <td>The Loft</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2015-02-06</td>
      <td>Jupiter Ascending</td>
      <td>Seventh Son</td>
      <td>SpongeBob Movie: Sponge Out of Water</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



For the purpose of plotting the frequency of an event, we are not interested in what movies were released, but only in how many on a particular date. We can count the movies by replacing the names with ones and calculating the sum.


```python
# replace movie names with ones
df.iloc[:,1:] = df.iloc[:,1:].replace(r'\w', 1.0, regex=True)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Film1</th>
      <th>Film2</th>
      <th>Film3</th>
      <th>Film4</th>
      <th>Film5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-09</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-16</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-01-23</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015-01-30</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2015-02-06</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We can finally get rid of the unnecessary columns and save the clean data to a file. Now we are ready to start plotting our event series data.


```python
df['film_sum'] =df[['Film1', 'Film2', 'Film3', 'Film4', 'Film5']].sum(axis=1)
df.drop(['Film1', 'Film2', 'Film3', 'Film4', 'Film5'], axis=1, inplace=True)
df.to_csv('data/filmrelease.csv')
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>film_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-09</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-16</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-01-23</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015-01-30</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2015-02-06</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



## Load the data to the plotting variables

To use this script, we have to load the clean data that we saved in the previuos steps.


```python
# Load the earthquake data and add a column with ones since there was only one earthquake per row
df1 = pd.read_csv('data/earthquakes.csv', header=None)
del df1[0]
df1.columns = ['Date']
df1['earthquake'] = np.ones(len(df1))
df1.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>earthquake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-12-09</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-12-08</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-12-08</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-12-06</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-11-25</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Load the movie data, the second column already shows us the sum of movie releases
df2 = pd.read_csv('data/filmrelease.csv', header=0)
del df2['Unnamed: 0']
df2.columns = ['Date', 'movie_release']
df2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>movie_release</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-09</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-16</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-23</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-30</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-06</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



As end result, we want to have a single DataFrame containing the data for both event series. Moreover, we want to have a continuous time series, including those days in which none of the two events took place (no earthquakes, and no movie releases). We do this by using the concatenate and resample functions of pandas.


```python
# Concatenate both DataFrames into one
df = pd.concat([df1, df2], ignore_index=True)
df = df.set_index(pd.DatetimeIndex(df.Date))
df = df.sort_index()
df = df.resample('1d').sum().fillna(0) # to complete every day
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>earthquake</th>
      <th>movie_release</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-09</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2015-01-10</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-01-11</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-01-12</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-01-13</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-01-14</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-01-15</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-01-16</th>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2015-01-17</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-01-18</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# Calculating and plotting a moving average

We could simply plot each event occurrence as a data point in a time series. However, this will likely yield a graph that is not very informative. Much easier to grasp is a moving average that tells us the average frequency of the events for a defined period of time in the past. We can create columns for these moving averages, which we can then easily plot.


```python
# Calculate moving average
for i in [7*4, 7*4*2]:
    mvav = i # moving average period, i.e. number of points to average
    dfi = np.convolve(df['earthquake'], 
                      np.ones((mvav,))*7/mvav # factor for obtaining average
                      , mode='full')
    df['earthquake moving average %sw' % (int(i/7))] = dfi[:-(i-1)]
    dfj = np.convolve(df['movie_release'], 
                      np.ones((mvav,))*7/mvav # factor for obtaining average
                      , mode='full')
    df['movie_release moving average %sw' % (int(i/7))] = dfj[:-(i-1)]
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>earthquake</th>
      <th>movie_release</th>
      <th>earthquake moving average 4w</th>
      <th>movie_release moving average 4w</th>
      <th>earthquake moving average 8w</th>
      <th>movie_release moving average 8w</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-09</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.125</td>
    </tr>
    <tr>
      <th>2015-01-10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.125</td>
    </tr>
    <tr>
      <th>2015-01-11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.125</td>
    </tr>
    <tr>
      <th>2015-01-12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.125</td>
    </tr>
    <tr>
      <th>2015-01-13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.125</td>
    </tr>
  </tbody>
</table>
</div>



The shorter the period that we choose for the moving average, the noisier our graphic will get. Let's settle with a moving average that reflects the frequency during the past 8 weeks. And _voilà_! Now we can see how the frequency of earthquakes and movie releases changed over time.


```python
# Plot relevant columns from dataframe
df.loc[:,['earthquake moving average 8w', 'movie_release moving average 8w']].\
plot(cmap='Set2', figsize=(12,4))# possible
plt.xlim(df.index[0], df.index.max()+10)
plt.title('Moving average of events per week')
plt.ylabel('Frequency')
plt.show()
```


    
![png](/images/2017-02-06-event-tracker_files/2017-02-06-event-tracker_29_0.png)
    


# Descriptive analysis of the event occurrence 

What other insights can we get from this data set? Two very dissimilar series of events, one natural, and one man-made, will surely have very different properties. Let's start with a simple question: on which days of the week do both events typically happen?


```python
#%% DAY OF THE WEEK ANALYSIS

# create column for day of the week
df['Day'] = df.index.dayofweek
df['Day'] = df.Day.astype('category')
df.Day.cat.categories = ['Mon','Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# create column for type
df['Type'] = np.where(df['earthquake']>0, 'earthquake', np.where(df['movie_release']>0,\
 'movie_release', np.nan))
df['Type'] = df['Type'].astype('category')
df['Type'] = df['Type'].cat.remove_categories(['nan'])

# show count for each day
#df[(df.Event1 == 1)&(df.Day == 'Mon')].Day.count()

# plot count data per day of the week
plt.figure()
plt.title('Event count per day of the week')
sns.countplot(data=df, x='Day', hue='Type')
sns.despine(left=True)
plt.show()
```


    
![png](/images/2017-02-06-event-tracker_files/2017-02-06-event-tracker_32_0.png)
    


We can see that nature does not respect our weekends, as earthquakes seem to be flatly distributed by day of the week.

The movie releases, on the other hand, are most frequent on Friday, followed by Wednesday and a few on Sunday (it's almost as if movie release days were _chosen_ by someone... /s). It seems that, if you're planning to release a new movie in the US, Friday is the way to go. People are usually happy to start the weekend with a leisurely activity, so that makes sense. As of why Saturdays and Sundays are almost not used as movie release days, even though on this days people are also usually free from work, it would be interesting to find out why. Another intriguing finding is the not high but remarkable number of releases on Wednesdays. Don't people work on Thursdays?

## Analysis of frequency per week

The world is a big place (or is it a small world?) and earthquakes occur all the time, even though we might not always find out. On the other hand, I would expect that movie releases occur much more frequently. So let's take a look at the data by plotting histograms for both events side by side.


```python
# joint histograms
plt.figure()
df['earthquake moving average 8w'].hist(alpha=.9)
df['movie_release moving average 8w'].hist(alpha=.9)
plt.title('Histogram of event frequency per week')
plt.show()

# With seaborn
sns.distplot(df['earthquake moving average 8w']), \
sns.distplot(df['movie_release moving average 8w'])
```


    
![png](/images/2017-02-06-event-tracker_files/2017-02-06-event-tracker_36_0.png)
    





    (<matplotlib.axes._subplots.AxesSubplot at 0x11eb49198>,
     <matplotlib.axes._subplots.AxesSubplot at 0x11eb49198>)




    
![png](/images/2017-02-06-event-tracker_files/2017-02-06-event-tracker_36_2.png)
    


Luckily, movie releases are much more frequent per week as earthquakes. On most weeks, there are between two and three movie releases, compared to 0.5 to 1.5 earthquakes.

# Final remarks

In this post, we gathered information on the occurrence of two events: earthquakes around the world, and movie releases in the US. By plotting their moving averages we could better compare when they occurred and gained some interesting insights about how they compare. All thanks to Python!


```python

```
