Learn how to summarize the columns available in an R data frame. 
  You will also learn how to chain operations together with the
  pipe operator, and how to compute grouped summaries using.

## Welcome!

Hey there! Ready for the first lesson?

The dfply package makes it possible to do R's dplyr-style data manipulation with pipes in python on pandas DataFrames.

[dfply website here](https://github.com/kieferk/dfply)

[![](https://www.rforecology.com/pipes_image0.png "https://github.com/kieferk/dfply"){width="600"}](https://github.com/kieferk/dfply)


```python
import pandas as pd
import seaborn as sns
cars = sns.load_dataset('mpg')
from dfply import *
cars >> head(3)
```

    Matplotlib is building the font cache; this may take a moment.
    


    ---------------------------------------------------------------------------

    gaierror                                  Traceback (most recent call last)

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\urllib\request.py:1348, in AbstractHTTPHandler.do_open(self, http_class, req, **http_conn_args)
       1347 try:
    -> 1348     h.request(req.get_method(), req.selector, req.data, headers,
       1349               encode_chunked=req.has_header('Transfer-encoding'))
       1350 except OSError as err: # timeout error
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\http\client.py:1282, in HTTPConnection.request(self, method, url, body, headers, encode_chunked)
       1281 """Send a complete request to the server."""
    -> 1282 self._send_request(method, url, body, headers, encode_chunked)
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\http\client.py:1328, in HTTPConnection._send_request(self, method, url, body, headers, encode_chunked)
       1327     body = _encode(body, 'body')
    -> 1328 self.endheaders(body, encode_chunked=encode_chunked)
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\http\client.py:1277, in HTTPConnection.endheaders(self, message_body, encode_chunked)
       1276     raise CannotSendHeader()
    -> 1277 self._send_output(message_body, encode_chunked=encode_chunked)
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\http\client.py:1037, in HTTPConnection._send_output(self, message_body, encode_chunked)
       1036 del self._buffer[:]
    -> 1037 self.send(msg)
       1039 if message_body is not None:
       1040 
       1041     # create a consistent interface to message_body
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\http\client.py:975, in HTTPConnection.send(self, data)
        974 if self.auto_open:
    --> 975     self.connect()
        976 else:
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\http\client.py:1447, in HTTPSConnection.connect(self)
       1445 "Connect to a host on a given (SSL) port."
    -> 1447 super().connect()
       1449 if self._tunnel_host:
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\http\client.py:941, in HTTPConnection.connect(self)
        940 sys.audit("http.client.connect", self, self.host, self.port)
    --> 941 self.sock = self._create_connection(
        942     (self.host,self.port), self.timeout, self.source_address)
        943 # Might fail in OSs that don't implement TCP_NODELAY
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\socket.py:824, in create_connection(address, timeout, source_address)
        823 err = None
    --> 824 for res in getaddrinfo(host, port, 0, SOCK_STREAM):
        825     af, socktype, proto, canonname, sa = res
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\socket.py:955, in getaddrinfo(host, port, family, type, proto, flags)
        954 addrlist = []
    --> 955 for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
        956     af, socktype, proto, canonname, sa = res
    

    gaierror: [Errno 11001] getaddrinfo failed

    
    During handling of the above exception, another exception occurred:
    

    URLError                                  Traceback (most recent call last)

    Cell In[2], line 3
          1 import pandas as pd
          2 import seaborn as sns
    ----> 3 cars = sns.load_dataset('mpg')
          4 from dfply import *
          5 cars >> head(3)
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\site-packages\seaborn\utils.py:572, in load_dataset(name, cache, data_home, **kws)
        570 cache_path = os.path.join(get_data_home(data_home), os.path.basename(url))
        571 if not os.path.exists(cache_path):
    --> 572     if name not in get_dataset_names():
        573         raise ValueError(f"'{name}' is not one of the example datasets.")
        574     urlretrieve(url, cache_path)
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\site-packages\seaborn\utils.py:499, in get_dataset_names()
        493 def get_dataset_names():
        494     """Report available example datasets, useful for reporting issues.
        495 
        496     Requires an internet connection.
        497 
        498     """
    --> 499     with urlopen(DATASET_NAMES_URL) as resp:
        500         txt = resp.read()
        502     dataset_names = [name.strip() for name in txt.decode().split("\n")]
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\urllib\request.py:216, in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        214 else:
        215     opener = _opener
    --> 216 return opener.open(url, data, timeout)
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\urllib\request.py:519, in OpenerDirector.open(self, fullurl, data, timeout)
        516     req = meth(req)
        518 sys.audit('urllib.Request', req.full_url, req.data, req.headers, req.get_method())
    --> 519 response = self._open(req, data)
        521 # post-process response
        522 meth_name = protocol+"_response"
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\urllib\request.py:536, in OpenerDirector._open(self, req, data)
        533     return result
        535 protocol = req.type
    --> 536 result = self._call_chain(self.handle_open, protocol, protocol +
        537                           '_open', req)
        538 if result:
        539     return result
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\urllib\request.py:496, in OpenerDirector._call_chain(self, chain, kind, meth_name, *args)
        494 for handler in handlers:
        495     func = getattr(handler, meth_name)
    --> 496     result = func(*args)
        497     if result is not None:
        498         return result
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\urllib\request.py:1391, in HTTPSHandler.https_open(self, req)
       1390 def https_open(self, req):
    -> 1391     return self.do_open(http.client.HTTPSConnection, req,
       1392         context=self._context, check_hostname=self._check_hostname)
    

    File c:\Users\bonat\AppData\Local\Programs\Python\Python310\lib\urllib\request.py:1351, in AbstractHTTPHandler.do_open(self, http_class, req, **http_conn_args)
       1348         h.request(req.get_method(), req.selector, req.data, headers,
       1349                   encode_chunked=req.has_header('Transfer-encoding'))
       1350     except OSError as err: # timeout error
    -> 1351         raise URLError(err)
       1352     r = h.getresponse()
       1353 except:
    

    URLError: <urlopen error [Errno 11001] getaddrinfo failed>


## The \>\> and \>\>=

dfply works directly on pandas DataFrames, chaining operations on the data with the >> operator, or alternatively starting with >>= for inplace operations.

*The X DataFrame symbol*

The DataFrame as it is passed through the piping operations is represented by the symbol X. It records the actions you want to take (represented by the Intention class), but does not evaluate them until the appropriate time. Operations on the DataFrame are deferred. Selecting two of the columns, for example, can be done using the symbolic X DataFrame during the piping operations.

### Exercise 1.

Select the columns 'mpg' and 'horsepower' from the cars DataFrame.


```python
cars >> select(X.mpg,X.horsepower) >> head(3)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 cars >> select(X.mpg,X.horsepower) >> head(3)
    

    NameError: name 'cars' is not defined


## Selecting and dropping

There are two functions for selection, inverse of each other: select and drop. The select and drop functions accept string labels, integer positions, and/or symbolically represented column names (X.column). They also accept symbolic "selection filter" functions, which will be covered shortly.

### Exercise 2.

Select the columns 'mpg' and 'horsepower' from the cars DataFrame using the drop function.


```python
cars >> drop(~X.horsepower, ~X.mpg) >> head(3)
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
      <th>mpg</th>
      <th>horsepower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>150.0</td>
    </tr>
  </tbody>
</table>
</div>



## Selection using \~

One particularly nice thing about dplyr's selection functions is that you can drop columns inside of a select statement by putting a subtraction sign in front, like so: ... %>% select(-col). The same can be done in dfply, but instead of the subtraction operator you use the tilde ~.

### Exercise 3.

Select all columns except 'model_year', and 'name' from the cars DataFrame.


```python
cars >> select(~X.model_year,~X.name) >> head(3)
```

## Filtering columns

The vanilla select and drop functions are useful, but there are a variety of selection functions inspired by dplyr available to make selecting and dropping columns a breeze. These functions are intended to be put inside of the select and drop functions, and can be paired with the ~ inverter.

First, a quick rundown of the available functions:

-   starts_with(prefix): find columns that start with a string prefix.
-   ends_with(suffix): find columns that end with a string suffix.
-   contains(substr): find columns that contain a substring in their name.
-   everything(): all columns.
-   columns_between(start_col, end_col, inclusive=True): find columns between a specified start and end column. The inclusive boolean keyword argument indicates whether the end column should be included or not.
-   columns_to(end_col, inclusive=True): get columns up to a specified end column. The inclusive argument indicates whether the ending column should be included or not.
-   columns_from(start_col): get the columns starting at a specified column.

### Exercise 4.

The selection filter functions are best explained by example. Let's say I wanted to select only the columns that started with a "c":


```python
cars >> select(starts_with("c")) >> head(3)
```

### Exercise 5.

Select the columns that contain the substring "e" from the cars DataFrame.


```python
cars >> select(contains("e"))>>head(3)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 cars >> select(contains("e"))>>head(3)
    

    NameError: name 'cars' is not defined


### Exercise 6.

Select the columns that are between 'mpg' and 'origin' from the cars DataFrame.


```python
cars >> select(columns_between("mpg", "origin")) >> head(3)
```

## Subsetting and filtering

### row_slice()

Slices of rows can be selected with the row_slice() function. You can pass single integer indices or a list of indices to select rows as with. This is going to be the same as using pandas' .iloc.

#### Exercise 7.

Select the first three rows from the cars DataFrame.


```python
cars >> row_slice([0,1,2])
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
    </tr>
  </tbody>
</table>
</div>



### distinct()

Selection of unique rows is done with distinct(), which similarly passes arguments and keyword arguments through to the DataFrame's .drop_duplicates() method.

#### Exercise 8.

Select the unique rows from the 'origin' column in the cars DataFrame.


```python
cars >> distinct(X.origin)
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>14</th>
      <td>24.0</td>
      <td>4</td>
      <td>113.0</td>
      <td>95.0</td>
      <td>2372</td>
      <td>15.0</td>
      <td>70</td>
      <td>japan</td>
      <td>toyota corona mark ii</td>
    </tr>
    <tr>
      <th>19</th>
      <td>26.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>46.0</td>
      <td>1835</td>
      <td>20.5</td>
      <td>70</td>
      <td>europe</td>
      <td>volkswagen 1131 deluxe sedan</td>
    </tr>
  </tbody>
</table>
</div>



## mask()

Filtering rows with logical criteria is done with mask(), which accepts boolean arrays "masking out" False labeled rows and keeping True labeled rows. These are best created with logical statements on symbolic Series objects as shown below. Multiple criteria can be supplied as arguments and their intersection will be used as the mask.

### Exercise 9.

Filter the cars DataFrame to only include rows where the 'mpg' is greater than 20, origin Japan, and display the first three rows:


```python
cars >> mask(X.mpg>20) >> head(3)
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>24.0</td>
      <td>4</td>
      <td>113.0</td>
      <td>95.0</td>
      <td>2372</td>
      <td>15.0</td>
      <td>70</td>
      <td>japan</td>
      <td>toyota corona mark ii</td>
    </tr>
    <tr>
      <th>15</th>
      <td>22.0</td>
      <td>6</td>
      <td>198.0</td>
      <td>95.0</td>
      <td>2833</td>
      <td>15.5</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth duster</td>
    </tr>
    <tr>
      <th>17</th>
      <td>21.0</td>
      <td>6</td>
      <td>200.0</td>
      <td>85.0</td>
      <td>2587</td>
      <td>16.0</td>
      <td>70</td>
      <td>usa</td>
      <td>ford maverick</td>
    </tr>
  </tbody>
</table>
</div>



## pull()

The pull() function is used to extract a single column from a DataFrame as a pandas Series. This is useful for passing a single column to a function or for further manipulation.

### Exercise 10.

Extract the 'mpg' column from the cars DataFrame, japanese origin, model year 70s, and display the first three rows.


```python
cars1 = cars >> mask(X.origin == 'japan', X.model_year == 70)
cars1['mpg']
```




    14    24.0
    18    27.0
    Name: mpg, dtype: float64



## DataFrame transformation

*mutate()*

The mutate() function is used to create new columns or modify existing columns. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 11.

Create a new column 'mpg_per_cylinder' in the cars DataFrame that is the result of dividing the 'mpg' column by the 'cylinders' column.


```python
cars >> mutate(mpg_per_cylinder=X.mpg/X.cylinders)>>head(3)
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
      <th>mpg_per_cylinder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>2.250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>1.875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
      <td>2.250</td>
    </tr>
  </tbody>
</table>
</div>




*transmute()*

The transmute() function is a combination of a mutate and a selection of the created variables.

### Exercise 12.

Create a new column 'mpg_per_cylinder' in the cars DataFrame that is the result of dividing the 'mpg' column by the 'cylinders' column, and display only the new column.


```python
cars >> transmute(mpg_per_cylinder=X.mpg/X.cylinders)>>head(3)
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
      <th>mpg_per_cylinder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.250</td>
    </tr>
  </tbody>
</table>
</div>



## Grouping

*group_by() and ungroup()*

The group_by() function is used to group the DataFrame by one or more columns. This is useful for creating groups of rows that can be summarized or transformed together. The ungroup() function is used to remove the grouping.

### Exercise 13.

Group the cars DataFrame by the 'origin' column and calculate the lead of the 'mpg' column.


```python
cars >> group_by(X.origin) >> mutate(mpg_lead=lead(X.mpg)) >> head(3)
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
      <th>mpg_lead</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>26.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>46.0</td>
      <td>1835</td>
      <td>20.5</td>
      <td>70</td>
      <td>europe</td>
      <td>volkswagen 1131 deluxe sedan</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>25.0</td>
      <td>4</td>
      <td>110.0</td>
      <td>87.0</td>
      <td>2672</td>
      <td>17.5</td>
      <td>70</td>
      <td>europe</td>
      <td>peugeot 504</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>24.0</td>
      <td>4</td>
      <td>107.0</td>
      <td>90.0</td>
      <td>2430</td>
      <td>14.5</td>
      <td>70</td>
      <td>europe</td>
      <td>audi 100 ls</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>24.0</td>
      <td>4</td>
      <td>113.0</td>
      <td>95.0</td>
      <td>2372</td>
      <td>15.0</td>
      <td>70</td>
      <td>japan</td>
      <td>toyota corona mark ii</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>27.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>88.0</td>
      <td>2130</td>
      <td>14.5</td>
      <td>70</td>
      <td>japan</td>
      <td>datsun pl510</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>27.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>88.0</td>
      <td>2130</td>
      <td>14.5</td>
      <td>71</td>
      <td>japan</td>
      <td>datsun pl510</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>



## Reshaping

*arrange()*

The arrange() function is used to sort the DataFrame by one or more columns. This is useful for reordering the rows of the DataFrame.

### Exercise 14.

Sort the cars DataFrame by the 'mpg' column in descending order.


```python
cars >> arrange(X.mpg, ascending = False) >> head(3)
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>322</th>
      <td>46.6</td>
      <td>4</td>
      <td>86.0</td>
      <td>65.0</td>
      <td>2110</td>
      <td>17.9</td>
      <td>80</td>
      <td>japan</td>
      <td>mazda glc</td>
    </tr>
    <tr>
      <th>329</th>
      <td>44.6</td>
      <td>4</td>
      <td>91.0</td>
      <td>67.0</td>
      <td>1850</td>
      <td>13.8</td>
      <td>80</td>
      <td>japan</td>
      <td>honda civic 1500 gl</td>
    </tr>
    <tr>
      <th>325</th>
      <td>44.3</td>
      <td>4</td>
      <td>90.0</td>
      <td>48.0</td>
      <td>2085</td>
      <td>21.7</td>
      <td>80</td>
      <td>europe</td>
      <td>vw rabbit c (diesel)</td>
    </tr>
  </tbody>
</table>
</div>




*rename()*

The rename() function is used to rename columns in the DataFrame. It accepts keyword arguments of the form new_column_name = old_column_name.

### Exercise 15.

Rename the 'mpg' column to 'miles_per_gallon' in the cars DataFrame.


```python
cars >> rename(miles_per_galon='mpg') >> head(3)
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
      <th>miles_per_galon</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
    </tr>
  </tbody>
</table>
</div>




*gather()*

The gather() function is used to reshape the DataFrame from wide to long format. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 16.

Reshape the cars DataFrame from wide to long format by gathering the columns 'mpg', 'horsepower', 'weight', 'acceleration', and 'displacement' into a new column 'variable' and their values into a new column 'value'.


```python
elonged = cars >> gather('variable', 'value', 'mpg', 'horsepower', 'weight', 'acceleration', 'displacement') >> group_by(X.variable)>>head(2)
elonged
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
      <th>cylinders</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1194</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>acceleration</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>1195</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>acceleration</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>1592</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>displacement</td>
      <td>307.0</td>
    </tr>
    <tr>
      <th>1593</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>displacement</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>398</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>horsepower</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>horsepower</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>mpg</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>mpg</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>796</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>weight</td>
      <td>3504.0</td>
    </tr>
    <tr>
      <th>797</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>weight</td>
      <td>3693.0</td>
    </tr>
  </tbody>
</table>
</div>




*spread()*

Likewise, you can transform a "long" DataFrame into a "wide" format with the spread(key, values) function. Converting the previously created elongated DataFrame for example would be done like so.

### Exercise 17.

Reshape the cars DataFrame from long to wide format by spreading the 'variable' column into columns and their values into the 'value' column.


```python
widened = elonged >> spread('variable', 'value')>>head(3)
widened
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
      <th>cylinders</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
      <th>acceleration</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>mpg</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>11.5</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>15.0</td>
      <td>3693.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>12.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>18.0</td>
      <td>3504.0</td>
    </tr>
  </tbody>
</table>
</div>




## Summarization

*summarize()*

The summarize() function is used to calculate summary statistics for groups of rows. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 18.

Calculate the mean 'mpg' for each group of 'origin' in the cars DataFrame.


```python
cars >> group_by(X.origin) >> summarize(mean_mpg=X.mpg.mean())
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
      <th>origin</th>
      <th>mean_mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>europe</td>
      <td>27.891429</td>
    </tr>
    <tr>
      <th>1</th>
      <td>japan</td>
      <td>30.450633</td>
    </tr>
    <tr>
      <th>2</th>
      <td>usa</td>
      <td>20.083534</td>
    </tr>
  </tbody>
</table>
</div>




*summarize_each()*

The summarize_each() function is used to calculate summary statistics for groups of rows. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 19.

Calculate the mean 'mpg' and 'horsepower' for each group of 'origin' in the cars DataFrame.


```python
cars >> group_by(X.origin)>>summarize_each([np.mean],X.mpg, X.horsepower)
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
      <th>origin</th>
      <th>mpg_mean</th>
      <th>horsepower_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>europe</td>
      <td>27.891429</td>
      <td>80.558824</td>
    </tr>
    <tr>
      <th>1</th>
      <td>japan</td>
      <td>30.450633</td>
      <td>79.835443</td>
    </tr>
    <tr>
      <th>2</th>
      <td>usa</td>
      <td>20.083534</td>
      <td>119.048980</td>
    </tr>
  </tbody>
</table>
</div>




*summarize() can of course be used with groupings as well.*

### Exercise 20.

Calculate the mean 'mpg' for each group of 'origin' and 'model_year' in the cars DataFrame.


```python
cars >> group_by(X.origin,X.model_year)>>summarize(mean_mpg=X.mpg.mean())
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
      <th>model_year</th>
      <th>origin</th>
      <th>mean_mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70</td>
      <td>europe</td>
      <td>25.200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71</td>
      <td>europe</td>
      <td>28.750000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>europe</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>73</td>
      <td>europe</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>europe</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>europe</td>
      <td>24.500000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>76</td>
      <td>europe</td>
      <td>24.250000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>77</td>
      <td>europe</td>
      <td>29.250000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78</td>
      <td>europe</td>
      <td>24.950000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>79</td>
      <td>europe</td>
      <td>30.450000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>80</td>
      <td>europe</td>
      <td>37.288889</td>
    </tr>
    <tr>
      <th>11</th>
      <td>81</td>
      <td>europe</td>
      <td>31.575000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>82</td>
      <td>europe</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>70</td>
      <td>japan</td>
      <td>25.500000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>71</td>
      <td>japan</td>
      <td>29.500000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>72</td>
      <td>japan</td>
      <td>24.200000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>73</td>
      <td>japan</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>74</td>
      <td>japan</td>
      <td>29.333333</td>
    </tr>
    <tr>
      <th>18</th>
      <td>75</td>
      <td>japan</td>
      <td>27.500000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>76</td>
      <td>japan</td>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>77</td>
      <td>japan</td>
      <td>27.416667</td>
    </tr>
    <tr>
      <th>21</th>
      <td>78</td>
      <td>japan</td>
      <td>29.687500</td>
    </tr>
    <tr>
      <th>22</th>
      <td>79</td>
      <td>japan</td>
      <td>32.950000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>80</td>
      <td>japan</td>
      <td>35.400000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>81</td>
      <td>japan</td>
      <td>32.958333</td>
    </tr>
    <tr>
      <th>25</th>
      <td>82</td>
      <td>japan</td>
      <td>34.888889</td>
    </tr>
    <tr>
      <th>26</th>
      <td>70</td>
      <td>usa</td>
      <td>15.272727</td>
    </tr>
    <tr>
      <th>27</th>
      <td>71</td>
      <td>usa</td>
      <td>18.100000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>72</td>
      <td>usa</td>
      <td>16.277778</td>
    </tr>
    <tr>
      <th>29</th>
      <td>73</td>
      <td>usa</td>
      <td>15.034483</td>
    </tr>
    <tr>
      <th>30</th>
      <td>74</td>
      <td>usa</td>
      <td>18.333333</td>
    </tr>
    <tr>
      <th>31</th>
      <td>75</td>
      <td>usa</td>
      <td>17.550000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>76</td>
      <td>usa</td>
      <td>19.431818</td>
    </tr>
    <tr>
      <th>33</th>
      <td>77</td>
      <td>usa</td>
      <td>20.722222</td>
    </tr>
    <tr>
      <th>34</th>
      <td>78</td>
      <td>usa</td>
      <td>21.772727</td>
    </tr>
    <tr>
      <th>35</th>
      <td>79</td>
      <td>usa</td>
      <td>23.478261</td>
    </tr>
    <tr>
      <th>36</th>
      <td>80</td>
      <td>usa</td>
      <td>25.914286</td>
    </tr>
    <tr>
      <th>37</th>
      <td>81</td>
      <td>usa</td>
      <td>27.530769</td>
    </tr>
    <tr>
      <th>38</th>
      <td>82</td>
      <td>usa</td>
      <td>29.450000</td>
    </tr>
  </tbody>
</table>
</div>


