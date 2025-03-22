# Exercise 3. - GroupBy

### Introduction:

GroupBy can be summarized as Split-Apply-Combine.

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out this [Diagram](http://i.imgur.com/yjNkiwL.png)  

Check out [Alcohol Consumption Exercises Video Tutorial](https://youtu.be/az67CMdmS6s) to watch a data scientist go through the exercises


### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv). 


```python
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv')
```

### Step 3. Assign it to a variable called drinks.


```python
drinks = df
drinks
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
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>AF</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>AF</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Venezuela</td>
      <td>333</td>
      <td>100</td>
      <td>3</td>
      <td>7.7</td>
      <td>SA</td>
    </tr>
    <tr>
      <th>189</th>
      <td>Vietnam</td>
      <td>111</td>
      <td>2</td>
      <td>1</td>
      <td>2.0</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>190</th>
      <td>Yemen</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.1</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>191</th>
      <td>Zambia</td>
      <td>32</td>
      <td>19</td>
      <td>4</td>
      <td>2.5</td>
      <td>AF</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Zimbabwe</td>
      <td>64</td>
      <td>18</td>
      <td>4</td>
      <td>4.7</td>
      <td>AF</td>
    </tr>
  </tbody>
</table>
<p>193 rows × 6 columns</p>
</div>



### Step 4. Which continent drinks more beer on average?


```python
drinks.groupby('continent').mean('beer_servings').sort_values('beer_servings', ascending=False).head(1)
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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EU</th>
      <td>193.777778</td>
      <td>132.555556</td>
      <td>142.222222</td>
      <td>8.617778</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. For each continent print the statistics for wine consumption.


```python
drinks.groupby('continent').sum('wine_servings')['wine_servings']
```




    continent
    AF     862
    AS     399
    EU    6400
    OC     570
    SA     749
    Name: wine_servings, dtype: int64



### Step 6. Print the mean alcohol consumption per continent for every column


```python
drinks.groupby('continent').mean(numeric_only=True)

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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>61.471698</td>
      <td>16.339623</td>
      <td>16.264151</td>
      <td>3.007547</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>37.045455</td>
      <td>60.840909</td>
      <td>9.068182</td>
      <td>2.170455</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>193.777778</td>
      <td>132.555556</td>
      <td>142.222222</td>
      <td>8.617778</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>89.687500</td>
      <td>58.437500</td>
      <td>35.625000</td>
      <td>3.381250</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>175.083333</td>
      <td>114.750000</td>
      <td>62.416667</td>
      <td>6.308333</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Print the median alcohol consumption per continent for every column


```python
drinks.groupby('continent').median(numeric_only=True)
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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>32.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>17.5</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>219.0</td>
      <td>122.0</td>
      <td>128.0</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>52.5</td>
      <td>37.0</td>
      <td>8.5</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>162.5</td>
      <td>108.5</td>
      <td>12.0</td>
      <td>6.85</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Print the mean, min and max values for spirit consumption.
#### This time output a DataFrame


```python
drinks['spirit_servings'].describe()
```




    count    193.000000
    mean      80.994819
    std       88.284312
    min        0.000000
    25%        4.000000
    50%       56.000000
    75%      128.000000
    max      438.000000
    Name: spirit_servings, dtype: float64


