# Streaming Anomaly Detection with Python-PySAD
![png](Headline.png)

## Introduction
This work belongs to Yuzy Ye, [Xinbo Wang](https://github.com/xinbo-hubert-wang), and Simin Liao as a team.

This project is meant to be an example of how to use the Python PySAD library to implement a machine learning based real-time anomaly detection system. The project builds on the NYC Taxi Open Data ([NYC TLC Open Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)) and demonstrates how this technique could help businesses capture unexpected fluctuations in their daily operations, and make timely responses to improve the business.

You can watch us explaining the project in this 5-min video on [YouTube](https://youtu.be/Wahh8MlUUpI)

You can also read a concise introduction to the topic in this [handout](https://github.com/xinbo-hubert-wang/PySAD-Anomaly-Detection-for-NYC-Taxi/blob/main/handout.pdf).

The rest of this doc and the repository will be focused on the technical details. If you are interesed in running the code your self, please turn to the [notebook](https://github.com/xinbo-hubert-wang/PySAD-Anomaly-Detection-for-NYC-Taxi/blob/main/PySAD-Anomaly-Detection-for-NYC-Taxi.ipynb) and download the prepared data provided in the repository.

## The Code Example

Following is the code example on how we can implement an anomaly detection system for NYC Taxi. 

The example includes three sections:
1. **Data Preparing**: We use pandas to read the data from NYC.org and transform them into the input for our data;
2. **Modelling with PySAD**: We use the PySAD package to build our Streaming Anomaly Detection (SAD) model for scoring the data points as we would do in actual production settings.
3. **Visualizing the Anomaly**: We picked a data point with high anomaly score (deemed as anomaly by SAD), and visualize the half-hour demand difference from normal expectations, so as to demonstrate the anomaly and provide insights on how to increase profits in face of this anomaly.

Each part of this erxample can be run individually with prepared data in the repository.

### 1. Data Preparing
The data we use is an open dataset provided by NYC.org.

In this section, we will start from acquiring raw data from NYC.org and try to summarize it down to the level where our model takes. This step resembles the data streaming and transforming process in actual implementation, but the data summarized each time then would be in much smaller size and therefore much quicker.

If you would like to run this notebook yourself, it is recommended that you skip this section and use the prepared dataset instead, since it would require you to download 7GB data from the website and will cost you a lot of time.


```python
import pandas as pd
```


```python
link = 'https://nyc-tlc.s3.amazonaws.com/trip+data/fhvhv_tripdata_{}.csv'
# we will be simulating the anomaly detection process in the time frame from 2019-11 to 2020-06
months = ['2019-11',
          '2019-12',
          '2020-01',
          '2020-02',
          '2020-03',
          '2020-04',
          '2020-05',
          '2020-06']
```


```python
# WARINING: again, if you are trying to run this notebook, it's recommended that you skip this
# and jump to the next section, since this chunk will download 7GB data and cost you a lot of time.
time_series = pd.DataFrame()

for month in [months[0]]:
    tmp = pd.read_csv(link.format(month), usecols=['pickup_datetime', 'PULocationID'])
    tmp['pickup_datetime'] = pd.to_datetime(tmp['pickup_datetime'])
    # this loop is to reduce memory requirement
    # this step also requires a lot of memory space
    dates = tmp['pickup_datetime'].dt.date.unique()
    for date in dates:
        print(date)
        df = tmp[tmp['pickup_datetime'].dt.date==date]
        df = pd.concat([df, pd.get_dummies(df['PULocationID'], prefix='zone')], axis=1)
        df = df.drop('PULocationID',axis=1).set_index('pickup_datetime')
        df = df.resample('30min').sum()
        time_series = pd.concat([time_series, df])
```

    2019-11-01
    2019-11-02
    2019-11-03
    2019-11-04
    2019-11-05
    2019-11-06
    2019-11-07
    2019-11-08
    2019-11-09
    2019-11-10
    2019-11-11
    2019-11-12
    2019-11-13
    2019-11-14
    2019-11-15
    2019-11-16
    2019-11-17
    2019-11-18
    2019-11-19
    2019-11-20
    2019-11-21
    2019-11-22
    2019-11-23
    2019-11-24
    2019-11-25
    2019-11-26
    2019-11-27
    2019-11-28
    2019-11-29
    2019-11-30
    


```python
time_series.to_csv('time_series_by_zone.csv')
```

### 2. Modelling with PySAD
This section will utilize the XStream model implemented by PySAD.


```python
import pandas as pd
from pysad.models import xStream
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
```


```python
time_series = pd.read_csv('time_series_by_zone.csv')
time_series['pickup_datetime'] = pd.to_datetime(time_series['pickup_datetime'])
time_series = time_series.set_index('pickup_datetime')
```

Since the taxi order through out the day following a similar pattern (same for weekly, monthly, yearly, etc.) The Seasonality in the fluctuation should be accounted (potentially use packages like prophet). 

In this case, however, we will mitigate this by compare data points with same time in day, for the sake of simplicity. You can always add other models to get an expected time series and use the difference in actual time series as the input.


```python
time_series['hour'] = time_series.index.hour
time_series['minute'] = time_series.index.minute
time_series['partition'] = time_series['hour'] * 2 + time_series['minute'] / 30 + 1

dataset = []

for i in range(1, 49):
    tmp = time_series[time_series['partition']==i].drop(['hour', 'minute', 'partition'], axis=1)
    dataset.append(tmp)
```

Training the model. This will take a long time since we are going through 1 million+ data points one by one, fitting and scoring in order.

In real-world implementation, it will only fit and score once for each data point that comes in, providing real-time feedback on anomaly.


```python
for data in dataset:
    # we estimate that 30 days is a good estimate for building an expectation on demands
    model = xStream(window_size = 30)
    calibrator = ConformalProbabilityCalibrator(windowed=True, window_size=30)
    scores = model.fit_score(data.values)
    scores = calibrator.fit_transform(scores)
    data['score'] = scores
```


```python
res = pd.DataFrame()

for data in dataset:
    res = pd.concat([res,data])
```


```python
res = res.sort_index()
res.head()
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
      <th>zone_1</th>
      <th>zone_2</th>
      <th>zone_3</th>
      <th>zone_4</th>
      <th>zone_5</th>
      <th>zone_6</th>
      <th>zone_7</th>
      <th>zone_8</th>
      <th>zone_9</th>
      <th>zone_10</th>
      <th>...</th>
      <th>zone_259</th>
      <th>zone_260</th>
      <th>zone_261</th>
      <th>zone_262</th>
      <th>zone_263</th>
      <th>zone_265</th>
      <th>zone_110</th>
      <th>zone_105</th>
      <th>zone_199</th>
      <th>score</th>
    </tr>
    <tr>
      <th>pickup_datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2019-11-01 00:00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>96.0</td>
      <td>4.0</td>
      <td>16.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>27.0</td>
      <td>...</td>
      <td>42.0</td>
      <td>61.0</td>
      <td>71.0</td>
      <td>27.0</td>
      <td>105.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2019-11-01 00:30:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>118.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>207.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>18.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>39.0</td>
      <td>51.0</td>
      <td>18.0</td>
      <td>60.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2019-11-01 01:00:00</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>97.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>162.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>39.0</td>
      <td>32.0</td>
      <td>20.0</td>
      <td>74.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2019-11-01 01:30:00</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>97.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>172.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>...</td>
      <td>30.0</td>
      <td>45.0</td>
      <td>41.0</td>
      <td>14.0</td>
      <td>43.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2019-11-01 02:00:00</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>66.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>121.0</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>17.0</td>
      <td>48.0</td>
      <td>19.0</td>
      <td>10.0</td>
      <td>41.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 263 columns</p>
</div>




```python
res.to_csv('res_by_half_hour_30.csv')
```

### Visualizing the Anomaly
In this section, we demonstrate an anomaly by visualize a demand change map at the time point. This is also where the insights come from. Managers and drivers of the company can use these insights in changing demands to adjust their normal strategies in chase of higher profits.


```python
import geopandas as gp
import pandas as pd
from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
```


```python
df = pd.read_csv('res_by_hour_30.csv')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df = pd.melt(df, 'pickup_datetime', df.columns[2:-2], 'LocationID', 'trips').fillna(0)

# The time point 2020-6-27 12:00:00 score high in anomaly.
# We try to visualize demands at that point compared to expected demands from previous 30 days 
time = datetime(2020,6,27,12,0)

now = df[df['pickup_datetime']==time]
earlier = df[(df['pickup_datetime'] >= time+timedelta(days=-30)) & 
             (df['pickup_datetime'] >= time+timedelta(days=-1))].groupby('LocationID')['trips'].mean()
earlier_std = df[(df['pickup_datetime'] >= time+timedelta(days=-30)) & 
             (df['pickup_datetime'] >= time+timedelta(days=-1))].groupby('LocationID')['trips'].std()
earlier.name = 'earlier'
earlier_std.name = 'earlier_std'
now = now.join(earlier, on = 'LocationID')
now = now.join(earlier_std, on = 'LocationID')
now['diff'] = now['trips'] - now['earlier']
# here we use z-score to describe the deviation from expectation
now['zscore'] = now['diff']/now['earlier_std']
# If the zone had low order counts in the previous, 
# it's subject to high volatility and is not really of interest
now.loc[now['earlier']<=10, 'zscore'] = 0
```


```python
# import and merge the shape file of the zones
m = gp.read_file('taxi_zones/taxi_zones.shp')
now['LocationID'] = now['LocationID'].apply(lambda x: int(x.split('_')[1]))
now = m.merge(now, on='LocationID')
```


```python
# visualize the change
fig = plt.figure(dpi=300,figsize=(10,14))
now.plot('zscore', ax = plt.gca(), cmap='RdBu', vmin=-2, vmax=2, linewidth=0.1, edgecolor='grey',
         legend=True, legend_kwds={'format': '%.1f', 'orientation': 'horizontal'})
plt.axis(False)
```




    (905464.7390389859, 1075092.878374982, 112485.76061678902, 280480.4146430247)




    
![png](output_20_1.png)
    


From the above map one can see increased demands in blue areas and decreased demands in red areas. 
