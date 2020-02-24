
# Mod 3 Final Project

## Student Info

- Name: Alex Beat
- Cohort: 100719 pt 
- Instructor: James Irving


## Instructions:

- Open and read the project assignment and guidelines in `MOD_PROJECT_README.ipynb`
- Review the hypothesis testing workflow found in this repo's `README.md` and at the bottom of the `MOD_PROJECT_README.ipynb`
- 3 functions from study group/learn.co lessons have been provided inside `functions.py`
    - `Cohen_d`, `find_outliers_IQR`,`find_outliers_Z`



<img src="https://raw.githubusercontent.com/jirvingphd/dsc-mod-3-project-online-ds-ft-100719/master/Northwind_ERD_updated.png">

# PROJECT


```python
!pip install -U fsds_100719
from fsds_100719.imports import *

import pandas as pd

```

    fsds_1007219  v0.7.6 loaded.  Read the docs: https://fsds.readthedocs.io/en/latest/ 



<style  type="text/css" >
</style><table id="T_6b98701c_55d3_11ea_9116_0026bb4edb26" ><caption>Loaded Packages and Handles</caption><thead>    <tr>        <th class="col_heading level0 col0" >Handle</th>        <th class="col_heading level0 col1" >Package</th>        <th class="col_heading level0 col2" >Description</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row0_col0" class="data row0 col0" >dp</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row0_col1" class="data row0 col1" >IPython.display</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row0_col2" class="data row0 col2" >Display modules with helpful display and clearing commands.</td>
            </tr>
            <tr>
                                <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row1_col0" class="data row1 col0" >fs</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row1_col1" class="data row1 col1" >fsds_100719</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row1_col2" class="data row1 col2" >Custom data science bootcamp student package</td>
            </tr>
            <tr>
                                <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row2_col0" class="data row2 col0" >mpl</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row2_col1" class="data row2 col1" >matplotlib</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row2_col2" class="data row2 col2" >Matplotlib's base OOP module with formatting artists</td>
            </tr>
            <tr>
                                <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row3_col0" class="data row3 col0" >plt</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row3_col1" class="data row3 col1" >matplotlib.pyplot</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row3_col2" class="data row3 col2" >Matplotlib's matlab-like plotting module</td>
            </tr>
            <tr>
                                <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row4_col0" class="data row4 col0" >np</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row4_col1" class="data row4 col1" >numpy</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row4_col2" class="data row4 col2" >scientific computing with Python</td>
            </tr>
            <tr>
                                <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row5_col0" class="data row5 col0" >pd</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row5_col1" class="data row5 col1" >pandas</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row5_col2" class="data row5 col2" >High performance data structures and tools</td>
            </tr>
            <tr>
                                <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row6_col0" class="data row6 col0" >sns</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row6_col1" class="data row6 col1" >seaborn</td>
                        <td id="T_6b98701c_55d3_11ea_9116_0026bb4edb26row6_col2" class="data row6 col2" >High-level data visualization library based on matplotlib</td>
            </tr>
    </tbody></table>



        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


    ['[i] Pandas .iplot() method activated.']



```python

import scipy.stats as stats
import numpy as np

def Cohen_d(group1, group2, correction = False):
    """Compute Cohen's d
    d = (group1.mean()-group2.mean())/pool_variance.
    pooled_variance= (n1 * var1 + n2 * var2) / (n1 + n2)

    Args:
        group1 (Series or NumPy array): group 1 for calculating d
        group2 (Series or NumPy array): group 2 for calculating d
        correction (bool): Apply equation correction if N<50. Default is False. 
            - Url with small ncorrection equation: 
                - https://www.statisticshowto.datasciencecentral.com/cohens-d/ 
    Returns:
        d (float): calculated d value
         
    INTERPRETATION OF COHEN's D: 
    > Small effect = 0.2
    > Medium Effect = 0.5
    > Large Effect = 0.8
    
    """
    import scipy.stats as stats
    import scipy   
    import numpy as np
    N = len(group1)+len(group2)
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    ## Apply correction if needed
    if (N < 50) & (correction==True):
        d=d * ((N-3)/(N-2.25))*np.sqrt((N-2)/N)
    return d


#Your code here
def find_outliers_Z(data):
    """Use scipy to calculate absolute Z-scores 
    and return boolean series where True indicates it is an outlier.

    Args:
        data (Series,or ndarray): data to test for outliers.

    Returns:
        [boolean Series]: A True/False for each row use to slice outliers.
        
    EXAMPLE USE: 
    >> idx_outs = find_outliers_df(df['AdjustedCompensation'])
    >> good_data = df[~idx_outs].copy()
    """
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import pandas as pd
    import numpy as np
    ## Calculate z-scores
    zs = stats.zscore(data)
    
    ## Find z-scores >3 awayfrom mean
    idx_outs = np.abs(zs)>3
    
    ## If input was a series, make idx_outs index match
    if isinstance(data,pd.Series):
        return pd.Series(idx_outs,index=data.index)
    else:
        return pd.Series(idx_outs)
    
    
    
def find_outliers_IQR(data):
    """Use Tukey's Method of outlier removal AKA InterQuartile-Range Rule
    and return boolean series where True indicates it is an outlier.
    - Calculates the range between the 75% and 25% quartiles
    - Outliers fall outside upper and lower limits, using a treshold of  1.5*IQR the 75% and 25% quartiles.

    IQR Range Calculation:    
        res = df.describe()
        IQR = res['75%'] -  res['25%']
        lower_limit = res['25%'] - 1.5*IQR
        upper_limit = res['75%'] + 1.5*IQR

    Args:
        data (Series,or ndarray): data to test for outliers.

    Returns:
        [boolean Series]: A True/False for each row use to slice outliers.
        
    EXAMPLE USE: 
    >> idx_outs = find_outliers_df(df['AdjustedCompensation'])
    >> good_data = df[~idx_outs].copy()
    
    """
    df_b=data
    res= df_b.describe()

    IQR = res['75%'] -  res['25%']
    lower_limit = res['25%'] - 1.5*IQR
    upper_limit = res['75%'] + 1.5*IQR

    idx_outs = (df_b>upper_limit) | (df_b<lower_limit)

    return idx_outs


def prep_data_for_tukeys(data):
    """Accepts a dictionary with group names as the keys 
    and pandas series as the values. 
    
    Returns a dataframe ready for tukeys test:
    - with a 'data' column and a 'group' column for sms.stats.multicomp.pairwise_tukeyhsd 
    
    Example Use:
    df_tukey = prep_data_for_tukeys(grp_data)
    tukey = sms.stats.multicomp.pairwise_tukeyhsd(df_tukey['data'], df_tukey['group'])
    tukey.summary()
    """
    import pandas as pd
    df_tukey = pd.DataFrame(columns=['data','group'])

    for k,v in  data.items():
        grp_df = v.rename('data').to_frame() 
        grp_df['group'] = k
        df_tukey=pd.concat([df_tukey,grp_df],axis=0)
    return df_tukey
```


```python
import sqlite3
connect = sqlite3.connect('Northwind_small.sqlite')
cur = connect.cursor()
```


```python
cur.execute("""SELECT name FROM sqlite_master WHERE type='table';""")
df_tables = pd.DataFrame(cur.fetchall(), columns=['Table'])
df_tables
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
      <th>Table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Employee</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Category</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Customer</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Shipper</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Supplier</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Order</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Product</td>
    </tr>
    <tr>
      <td>7</td>
      <td>OrderDetail</td>
    </tr>
    <tr>
      <td>8</td>
      <td>CustomerCustomerDemo</td>
    </tr>
    <tr>
      <td>9</td>
      <td>CustomerDemographic</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Region</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Territory</td>
    </tr>
    <tr>
      <td>12</td>
      <td>EmployeeTerritory</td>
    </tr>
  </tbody>
</table>
</div>



# Hypothesis 1

## Question

> Does discount amount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount? 

## Null and Alternative Hypothesis

- $H_0$: Customers bought the same quantity of both discounted and non-discounted products. 

- $H_1$: Customers bought the same quantity of both discounted and non-discounted products.

## STEP 1: Determine the category/type of test based on your data

Type of data: numeric <br>
    How many groups compared: Two different groups if comparing discount or full price.
        - So we use 2 sample t-test

### split data into 2 groups


```python
cur.execute("""SELECT * from OrderDetail""")
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id', 'OrderId', 'ProductId', 'UnitPrice', 'Quantity', 'Discount']




```python
h1df = pd.DataFrame(cur.fetchall(), columns=col_names)
h1df
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
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.00</td>
      <td>12</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.80</td>
      <td>10</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.80</td>
      <td>5</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.60</td>
      <td>9</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.40</td>
      <td>40</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>11077</td>
      <td>64</td>
      <td>33.25</td>
      <td>2</td>
      <td>0.03</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>11077</td>
      <td>66</td>
      <td>17.00</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>11077</td>
      <td>73</td>
      <td>15.00</td>
      <td>2</td>
      <td>0.01</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>11077</td>
      <td>75</td>
      <td>7.75</td>
      <td>4</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>11077</td>
      <td>77</td>
      <td>13.00</td>
      <td>2</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 6 columns</p>
</div>




```python
## Create 'discounted' column for groupby 
h1df['discounted'] = h1df['Discount']>0
h1df['discounted'] = h1df['discounted'].map({True:'Discounted',False:'Full Price'})
h1df
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
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.00</td>
      <td>12</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.80</td>
      <td>10</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.80</td>
      <td>5</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.60</td>
      <td>9</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.40</td>
      <td>40</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>11077</td>
      <td>64</td>
      <td>33.25</td>
      <td>2</td>
      <td>0.03</td>
      <td>Discounted</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>11077</td>
      <td>66</td>
      <td>17.00</td>
      <td>1</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>11077</td>
      <td>73</td>
      <td>15.00</td>
      <td>2</td>
      <td>0.01</td>
      <td>Discounted</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>11077</td>
      <td>75</td>
      <td>7.75</td>
      <td>4</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>11077</td>
      <td>77</td>
      <td>13.00</td>
      <td>2</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 7 columns</p>
</div>




```python
sns.barplot(data=h1df, x='discounted',y='Quantity',ci=68,)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2595ee48>




![png](output_22_1.png)


### put data into dict


```python
h1dict = {}
for name in h1df['discounted'].unique():
    h1dict[name] = h1df.groupby('discounted').get_group(name)['Quantity']
    
h1dict
```




    {'Full Price': 0       12
     1       10
     2        5
     3        9
     4       40
             ..
     2147     2
     2148     2
     2151     1
     2153     4
     2154     2
     Name: Quantity, Length: 1317, dtype: int64, 'Discounted': 6       35
     7       15
     8        6
     9       15
     11      40
             ..
     2144     2
     2146     3
     2149     2
     2150     2
     2152     2
     Name: Quantity, Length: 838, dtype: int64}



## STEP 2: Do we meet the assumptions of the chosen test?

Independent t-test (2-sample)

- No significant outliers
- Normality
- Equal Variance

### 0. Check for & Remove Outliers


```python
##Outlier Removal
fig,ax=plt.subplots(figsize=(8,4))
for name,values in h1dict.items():
    sns.distplot(values,label=name, ax=ax)
    
    
ax.legend()
ax.set(title='Quantity for Full Price vs Discounted Products', ylabel='Density')
```




    [Text(0, 0.5, 'Density'),
     Text(0.5, 1.0, 'Quantity for Full Price vs Discounted Products')]




![png](output_28_1.png)



```python
for name,values in h1dict.items():
    idx_outs = find_outliers_Z(values)
    print(f"Found {idx_outs.sum()} outliers using Z-score method for {name}.")
    h1dict[name] = values[~idx_outs]
    
```

    Found 20 outliers using Z-score method for Full Price.
    Found 15 outliers using Z-score method for Discounted.


### Check result of outlier removal on plot


```python
fig,ax=plt.subplots(figsize=(8,5))
for name,values in h1dict.items():
    sns.distplot(values,label=name, ax=ax)
    
    
ax.legend()
ax.set(title='Quantity for Full Price vs Discounted Products', ylabel='Density')
```




    [Text(0, 0.5, 'Density'),
     Text(0.5, 1.0, 'Quantity for Full Price vs Discounted Products')]




![png](output_31_1.png)


### 1. Test Assumption of Normality


```python
import scipy.stats as stats
```


```python
for key,values in h1dict.items():
    stat,p = stats.normaltest(values)
    print(f"Group {key} Normaltest p-value={round(p,4)}")
    sig = 'is NOT' if p<.05 else 'IS'

    print(f"\t-The data {sig} normal.")
```

    Group Full Price Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Discounted Normaltest p-value=0.0
    	-The data is NOT normal.


### 1B. We don't have normal data:


```python
len(h1dict['Full Price']), len(h1dict['Discounted'])
```




    (1297, 823)



But group sizes (n) are bigger than required 15 for each group so we can safely ignore normality assumption. <br>
So we move on to equal variance assumption.

### 2. Test for Equal Variance


```python
h1data = []
for key,vals in h1dict.items():
    h1data.append(vals)
```


```python
h1data[0]
```




    0       12
    1       10
    2        5
    3        9
    4       40
            ..
    2147     2
    2148     2
    2151     1
    2153     4
    2154     2
    Name: Quantity, Length: 1297, dtype: int64




```python
stats.levene(h1data[0],h1data[1])
```




    LeveneResult(statistic=19.187113832590878, pvalue=1.2429073348187694e-05)




```python
stat,p = stats.levene(*h1data)
print(f"Levene's Test for Equal Variance p-value={round(p,4)}")
sig = 'do NOT' if p<.05 else 'DO'

print(f"\t-The groups {sig} have equal variance.")
```

    Levene's Test for Equal Variance p-value=0.0
    	-The groups do NOT have equal variance.


### Failed the assumption of equal variance:

Since we don't have equal variance and data is not normal, we need to use a non-parametric version of a t-test. 
 - The non-parametric version for 2 sample is Mann-Whitney U test.
 - Works from medians instead of means. 
 - scipy.stats.mannwhitneyu()

## STEP 3: Interpret Result & Post-Hoc Tests

### Perform hypothesis test from summary table above to get your p-value.

- If p value is < $\alpha$:

    - Reject the null hypothesis.
    - Calculate effect size (e.g. Cohen's $d$)
- If p<.05 AND you have multiple groups (i.e. ANOVA)

    - Must run a pairwise Tukey's test to know which groups were significantly different.
    - Tukey pairwise comparison test
    - statsmodels.stats.multicomp.pairwise_tukeyhsd
- Report statistical power (optional)


```python
stat,h1p = stats.mannwhitneyu(h1data[0],h1data[1])
print(f"Mann Whitney U p-value={round(h1p,4)}")
```

    Mann Whitney U p-value=0.0


Mann Whitney p val defaults to one-sided test, so double that p val for our two sided test, and we will reject null hypothesis. 

### Calculate effect size (e.g. Cohen's  𝑑 )


```python
h1effectsize = Cohen_d(h1dict['Full Price'],h1dict['Discounted'])
h1effectsize
```




    -0.32001140965727837



0.3 effect size is small but still holds up. 

### Report statistical power (optional)

statsmodels.stats.power:
TTestIndPower , TTestPower


```python
from statsmodels.stats.power import TTestIndPower, TTestPower
power_analysis = TTestIndPower()
```


```python
len(h1dict['Full Price']+h1dict['Discounted'])
```




    2120




```python
# show the power to see how well we can reject null.
power_analysis.solve_power(effect_size=h1effectsize, nobs1=2120, alpha=h1p)
```




    0.9999327217471883




```python
# power is .99 so yeah we have a solid bet of rejecting null hypothesis and not getting any type 1 errors.
```

## What levels of discount?

> Does discount amount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount? 

### Make dict into dataframes for tukeys


```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

```


```python
h1df.head()
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
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>Full Price</td>
    </tr>
  </tbody>
</table>
</div>




```python
# -- Side note, this tukeys test is using original df containing all outliers.
h1tukey = pairwise_tukeyhsd(h1df['Quantity'],h1df['Discount'])
h1tukey.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD, FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>  <th>p-adj</th>   <th>lower</th>   <th>upper</th>  <th>reject</th>
</tr>
<tr>
    <td>0.0</td>   <td>0.01</td>  <td>-19.7153</td>   <td>0.9</td>  <td>-80.3306</td> <td>40.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.02</td>  <td>-19.7153</td>   <td>0.9</td>   <td>-62.593</td> <td>23.1625</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.03</td>  <td>-20.0486</td>  <td>0.725</td> <td>-55.0714</td> <td>14.9742</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.04</td>  <td>-20.7153</td>   <td>0.9</td>  <td>-81.3306</td> <td>39.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.05</td>   <td>6.2955</td>  <td>0.0011</td>  <td>1.5381</td>  <td>11.053</td>   <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.06</td>  <td>-19.7153</td>   <td>0.9</td>  <td>-80.3306</td> <td>40.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.1</td>   <td>3.5217</td>  <td>0.4269</td>  <td>-1.3783</td> <td>8.4217</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.15</td>   <td>6.6669</td>  <td>0.0014</td>   <td>1.551</td>  <td>11.7828</td>  <td>True</td> 
</tr>
<tr>
    <td>0.0</td>    <td>0.2</td>   <td>5.3096</td>  <td>0.0303</td>  <td>0.2508</td>  <td>10.3684</td>  <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.25</td>    <td>6.525</td>  <td>0.0023</td>  <td>1.3647</td>  <td>11.6852</td>  <td>True</td> 
</tr>
<tr>
   <td>0.01</td>   <td>0.02</td>     <td>0.0</td>     <td>0.9</td>  <td>-74.2101</td> <td>74.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.03</td>   <td>-0.3333</td>   <td>0.9</td>  <td>-70.2993</td> <td>69.6326</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.04</td>    <td>-1.0</td>     <td>0.9</td>  <td>-86.6905</td> <td>84.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.05</td>   <td>26.0108</td>   <td>0.9</td>   <td>-34.745</td> <td>86.7667</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.06</td>     <td>0.0</td>     <td>0.9</td>  <td>-85.6905</td> <td>85.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.1</td>   <td>23.237</td>    <td>0.9</td>  <td>-37.5302</td> <td>84.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.15</td>   <td>26.3822</td>   <td>0.9</td>  <td>-34.4028</td> <td>87.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.2</td>   <td>25.0248</td>   <td>0.9</td>  <td>-35.7554</td> <td>85.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.25</td>   <td>26.2403</td>   <td>0.9</td>  <td>-34.5485</td> <td>87.029</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.03</td>   <td>-0.3333</td>   <td>0.9</td>  <td>-55.6463</td> <td>54.9796</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.04</td>    <td>-1.0</td>     <td>0.9</td>  <td>-75.2101</td> <td>73.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.05</td>   <td>26.0108</td> <td>0.6622</td> <td>-17.0654</td> <td>69.087</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.06</td>     <td>0.0</td>     <td>0.9</td>  <td>-74.2101</td> <td>74.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.1</td>   <td>23.237</td>  <td>0.7914</td> <td>-19.8552</td> <td>66.3292</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.15</td>   <td>26.3822</td> <td>0.6461</td> <td>-16.7351</td> <td>69.4994</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.2</td>   <td>25.0248</td> <td>0.7089</td> <td>-18.0857</td> <td>68.1354</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.25</td>   <td>26.2403</td> <td>0.6528</td> <td>-16.8823</td> <td>69.3628</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.04</td>   <td>-0.6667</td>   <td>0.9</td>  <td>-70.6326</td> <td>69.2993</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.05</td>   <td>26.3441</td> <td>0.3639</td>  <td>-8.9214</td> <td>61.6096</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.06</td>   <td>0.3333</td>    <td>0.9</td>  <td>-69.6326</td> <td>70.2993</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.1</td>   <td>23.5703</td> <td>0.5338</td> <td>-11.7147</td> <td>58.8553</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.15</td>   <td>26.7155</td> <td>0.3436</td>  <td>-8.6001</td> <td>62.0311</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.2</td>   <td>25.3582</td>  <td>0.428</td>  <td>-9.9492</td> <td>60.6656</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.25</td>   <td>26.5736</td> <td>0.3525</td>  <td>-8.7485</td> <td>61.8957</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.05</td>   <td>27.0108</td>   <td>0.9</td>   <td>-33.745</td> <td>87.7667</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.06</td>     <td>1.0</td>     <td>0.9</td>  <td>-84.6905</td> <td>86.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.1</td>   <td>24.237</td>    <td>0.9</td>  <td>-36.5302</td> <td>85.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.15</td>   <td>27.3822</td>   <td>0.9</td>  <td>-33.4028</td> <td>88.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.2</td>   <td>26.0248</td>   <td>0.9</td>  <td>-34.7554</td> <td>86.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.25</td>   <td>27.2403</td>   <td>0.9</td>  <td>-33.5485</td> <td>88.029</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.06</td>  <td>-26.0108</td>   <td>0.9</td>  <td>-86.7667</td> <td>34.745</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.1</td>   <td>-2.7738</td>   <td>0.9</td>   <td>-9.1822</td> <td>3.6346</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.15</td>   <td>0.3714</td>    <td>0.9</td>   <td>-6.2036</td> <td>6.9463</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.2</td>   <td>-0.986</td>    <td>0.9</td>   <td>-7.5166</td> <td>5.5447</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.25</td>   <td>0.2294</td>    <td>0.9</td>   <td>-6.3801</td>  <td>6.839</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.1</td>   <td>23.237</td>    <td>0.9</td>  <td>-37.5302</td> <td>84.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.15</td>   <td>26.3822</td>   <td>0.9</td>  <td>-34.4028</td> <td>87.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.2</td>   <td>25.0248</td>   <td>0.9</td>  <td>-35.7554</td> <td>85.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.25</td>   <td>26.2403</td>   <td>0.9</td>  <td>-34.5485</td> <td>87.029</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.15</td>   <td>3.1452</td>    <td>0.9</td>   <td>-3.5337</td>  <td>9.824</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>    <td>0.2</td>   <td>1.7879</td>    <td>0.9</td>   <td>-4.8474</td> <td>8.4231</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.25</td>   <td>3.0033</td>    <td>0.9</td>   <td>-3.7096</td> <td>9.7161</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>    <td>0.2</td>   <td>-1.3573</td>   <td>0.9</td>   <td>-8.1536</td> <td>5.4389</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>   <td>0.25</td>   <td>-0.1419</td>   <td>0.9</td>   <td>-7.014</td>  <td>6.7302</td>   <td>False</td>
</tr>
<tr>
    <td>0.2</td>   <td>0.25</td>   <td>1.2154</td>    <td>0.9</td>   <td>-5.6143</td> <td>8.0451</td>   <td>False</td>
</tr>
</table>




```python
h1_tukeydf = pd.DataFrame(data=h1tukey._results_table.data[1:], columns=h1tukey._results_table.data[0])
h1_tukeydf
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
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>-19.7153</td>
      <td>0.9000</td>
      <td>-80.3306</td>
      <td>40.9001</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>-19.7153</td>
      <td>0.9000</td>
      <td>-62.5930</td>
      <td>23.1625</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>-20.0486</td>
      <td>0.7250</td>
      <td>-55.0714</td>
      <td>14.9742</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>-20.7153</td>
      <td>0.9000</td>
      <td>-81.3306</td>
      <td>39.9001</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>6.2955</td>
      <td>0.0011</td>
      <td>1.5381</td>
      <td>11.0530</td>
      <td>True</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>-19.7153</td>
      <td>0.9000</td>
      <td>-80.3306</td>
      <td>40.9001</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.00</td>
      <td>0.10</td>
      <td>3.5217</td>
      <td>0.4269</td>
      <td>-1.3783</td>
      <td>8.4217</td>
      <td>False</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.00</td>
      <td>0.15</td>
      <td>6.6669</td>
      <td>0.0014</td>
      <td>1.5510</td>
      <td>11.7828</td>
      <td>True</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.00</td>
      <td>0.20</td>
      <td>5.3096</td>
      <td>0.0303</td>
      <td>0.2508</td>
      <td>10.3684</td>
      <td>True</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>6.5250</td>
      <td>0.0023</td>
      <td>1.3647</td>
      <td>11.6852</td>
      <td>True</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.0000</td>
      <td>0.9000</td>
      <td>-74.2101</td>
      <td>74.2101</td>
      <td>False</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.01</td>
      <td>0.03</td>
      <td>-0.3333</td>
      <td>0.9000</td>
      <td>-70.2993</td>
      <td>69.6326</td>
      <td>False</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>-1.0000</td>
      <td>0.9000</td>
      <td>-86.6905</td>
      <td>84.6905</td>
      <td>False</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>26.0108</td>
      <td>0.9000</td>
      <td>-34.7450</td>
      <td>86.7667</td>
      <td>False</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.01</td>
      <td>0.06</td>
      <td>0.0000</td>
      <td>0.9000</td>
      <td>-85.6905</td>
      <td>85.6905</td>
      <td>False</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.01</td>
      <td>0.10</td>
      <td>23.2370</td>
      <td>0.9000</td>
      <td>-37.5302</td>
      <td>84.0042</td>
      <td>False</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.01</td>
      <td>0.15</td>
      <td>26.3822</td>
      <td>0.9000</td>
      <td>-34.4028</td>
      <td>87.1671</td>
      <td>False</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.01</td>
      <td>0.20</td>
      <td>25.0248</td>
      <td>0.9000</td>
      <td>-35.7554</td>
      <td>85.8050</td>
      <td>False</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.01</td>
      <td>0.25</td>
      <td>26.2403</td>
      <td>0.9000</td>
      <td>-34.5485</td>
      <td>87.0290</td>
      <td>False</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>-0.3333</td>
      <td>0.9000</td>
      <td>-55.6463</td>
      <td>54.9796</td>
      <td>False</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>-1.0000</td>
      <td>0.9000</td>
      <td>-75.2101</td>
      <td>73.2101</td>
      <td>False</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>26.0108</td>
      <td>0.6622</td>
      <td>-17.0654</td>
      <td>69.0870</td>
      <td>False</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.02</td>
      <td>0.06</td>
      <td>0.0000</td>
      <td>0.9000</td>
      <td>-74.2101</td>
      <td>74.2101</td>
      <td>False</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.02</td>
      <td>0.10</td>
      <td>23.2370</td>
      <td>0.7914</td>
      <td>-19.8552</td>
      <td>66.3292</td>
      <td>False</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.02</td>
      <td>0.15</td>
      <td>26.3822</td>
      <td>0.6461</td>
      <td>-16.7351</td>
      <td>69.4994</td>
      <td>False</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.02</td>
      <td>0.20</td>
      <td>25.0248</td>
      <td>0.7089</td>
      <td>-18.0857</td>
      <td>68.1354</td>
      <td>False</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.02</td>
      <td>0.25</td>
      <td>26.2403</td>
      <td>0.6528</td>
      <td>-16.8823</td>
      <td>69.3628</td>
      <td>False</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>-0.6667</td>
      <td>0.9000</td>
      <td>-70.6326</td>
      <td>69.2993</td>
      <td>False</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>26.3441</td>
      <td>0.3639</td>
      <td>-8.9214</td>
      <td>61.6096</td>
      <td>False</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.03</td>
      <td>0.06</td>
      <td>0.3333</td>
      <td>0.9000</td>
      <td>-69.6326</td>
      <td>70.2993</td>
      <td>False</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.03</td>
      <td>0.10</td>
      <td>23.5703</td>
      <td>0.5338</td>
      <td>-11.7147</td>
      <td>58.8553</td>
      <td>False</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.03</td>
      <td>0.15</td>
      <td>26.7155</td>
      <td>0.3436</td>
      <td>-8.6001</td>
      <td>62.0311</td>
      <td>False</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.03</td>
      <td>0.20</td>
      <td>25.3582</td>
      <td>0.4280</td>
      <td>-9.9492</td>
      <td>60.6656</td>
      <td>False</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.03</td>
      <td>0.25</td>
      <td>26.5736</td>
      <td>0.3525</td>
      <td>-8.7485</td>
      <td>61.8957</td>
      <td>False</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>27.0108</td>
      <td>0.9000</td>
      <td>-33.7450</td>
      <td>87.7667</td>
      <td>False</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.04</td>
      <td>0.06</td>
      <td>1.0000</td>
      <td>0.9000</td>
      <td>-84.6905</td>
      <td>86.6905</td>
      <td>False</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.04</td>
      <td>0.10</td>
      <td>24.2370</td>
      <td>0.9000</td>
      <td>-36.5302</td>
      <td>85.0042</td>
      <td>False</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.04</td>
      <td>0.15</td>
      <td>27.3822</td>
      <td>0.9000</td>
      <td>-33.4028</td>
      <td>88.1671</td>
      <td>False</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.04</td>
      <td>0.20</td>
      <td>26.0248</td>
      <td>0.9000</td>
      <td>-34.7554</td>
      <td>86.8050</td>
      <td>False</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.04</td>
      <td>0.25</td>
      <td>27.2403</td>
      <td>0.9000</td>
      <td>-33.5485</td>
      <td>88.0290</td>
      <td>False</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.05</td>
      <td>0.06</td>
      <td>-26.0108</td>
      <td>0.9000</td>
      <td>-86.7667</td>
      <td>34.7450</td>
      <td>False</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0.05</td>
      <td>0.10</td>
      <td>-2.7738</td>
      <td>0.9000</td>
      <td>-9.1822</td>
      <td>3.6346</td>
      <td>False</td>
    </tr>
    <tr>
      <td>42</td>
      <td>0.05</td>
      <td>0.15</td>
      <td>0.3714</td>
      <td>0.9000</td>
      <td>-6.2036</td>
      <td>6.9463</td>
      <td>False</td>
    </tr>
    <tr>
      <td>43</td>
      <td>0.05</td>
      <td>0.20</td>
      <td>-0.9860</td>
      <td>0.9000</td>
      <td>-7.5166</td>
      <td>5.5447</td>
      <td>False</td>
    </tr>
    <tr>
      <td>44</td>
      <td>0.05</td>
      <td>0.25</td>
      <td>0.2294</td>
      <td>0.9000</td>
      <td>-6.3801</td>
      <td>6.8390</td>
      <td>False</td>
    </tr>
    <tr>
      <td>45</td>
      <td>0.06</td>
      <td>0.10</td>
      <td>23.2370</td>
      <td>0.9000</td>
      <td>-37.5302</td>
      <td>84.0042</td>
      <td>False</td>
    </tr>
    <tr>
      <td>46</td>
      <td>0.06</td>
      <td>0.15</td>
      <td>26.3822</td>
      <td>0.9000</td>
      <td>-34.4028</td>
      <td>87.1671</td>
      <td>False</td>
    </tr>
    <tr>
      <td>47</td>
      <td>0.06</td>
      <td>0.20</td>
      <td>25.0248</td>
      <td>0.9000</td>
      <td>-35.7554</td>
      <td>85.8050</td>
      <td>False</td>
    </tr>
    <tr>
      <td>48</td>
      <td>0.06</td>
      <td>0.25</td>
      <td>26.2403</td>
      <td>0.9000</td>
      <td>-34.5485</td>
      <td>87.0290</td>
      <td>False</td>
    </tr>
    <tr>
      <td>49</td>
      <td>0.10</td>
      <td>0.15</td>
      <td>3.1452</td>
      <td>0.9000</td>
      <td>-3.5337</td>
      <td>9.8240</td>
      <td>False</td>
    </tr>
    <tr>
      <td>50</td>
      <td>0.10</td>
      <td>0.20</td>
      <td>1.7879</td>
      <td>0.9000</td>
      <td>-4.8474</td>
      <td>8.4231</td>
      <td>False</td>
    </tr>
    <tr>
      <td>51</td>
      <td>0.10</td>
      <td>0.25</td>
      <td>3.0033</td>
      <td>0.9000</td>
      <td>-3.7096</td>
      <td>9.7161</td>
      <td>False</td>
    </tr>
    <tr>
      <td>52</td>
      <td>0.15</td>
      <td>0.20</td>
      <td>-1.3573</td>
      <td>0.9000</td>
      <td>-8.1536</td>
      <td>5.4389</td>
      <td>False</td>
    </tr>
    <tr>
      <td>53</td>
      <td>0.15</td>
      <td>0.25</td>
      <td>-0.1419</td>
      <td>0.9000</td>
      <td>-7.0140</td>
      <td>6.7302</td>
      <td>False</td>
    </tr>
    <tr>
      <td>54</td>
      <td>0.20</td>
      <td>0.25</td>
      <td>1.2154</td>
      <td>0.9000</td>
      <td>-5.6143</td>
      <td>8.0451</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
h1_tukeydf.loc[h1_tukeydf['reject']==True]
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
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>6.2955</td>
      <td>0.0011</td>
      <td>1.5381</td>
      <td>11.0530</td>
      <td>True</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.0</td>
      <td>0.15</td>
      <td>6.6669</td>
      <td>0.0014</td>
      <td>1.5510</td>
      <td>11.7828</td>
      <td>True</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.0</td>
      <td>0.20</td>
      <td>5.3096</td>
      <td>0.0303</td>
      <td>0.2508</td>
      <td>10.3684</td>
      <td>True</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>6.5250</td>
      <td>0.0023</td>
      <td>1.3647</td>
      <td>11.6852</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



So it looks like the 5%, 15%, 20% and 25% discounts were all significant in effecting the average quantity sold per order. 


```python
# contains all outliers prior to outlier removal

sns.barplot(data=h1df,x='Discount',y='Quantity',ci=68)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a25cf22e8>




![png](output_67_1.png)


# Hypothesis 2

## Question

> Does discount amount have a statistically significant effect on the total spent ($) on a product in an order? If so, at what level(s) of discount?

## Null and Alternative Hypothesis

- $H_0$: Customers spend the same total amounts on both discounted and non-discounted products. 

- $H_1$: Customers spend different total amounts of money on discounted products vs full price products. 

## STEP 1: Determine the category/type of test based on your data

Type of data: numeric <br>
    How many groups compared: Two different groups if comparing discount or full price.
        - So we use 2 sample t-test

### split data into 2 groups


```python
cur.execute("""SELECT Id, UnitPrice, Quantity, Discount from OrderDetail""")
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id', 'UnitPrice', 'Quantity', 'Discount']




```python
h2df = pd.DataFrame(cur.fetchall(), columns=col_names)
h2df
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
      <th>Id</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>14.00</td>
      <td>12</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>9.80</td>
      <td>10</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>34.80</td>
      <td>5</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>18.60</td>
      <td>9</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>42.40</td>
      <td>40</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>33.25</td>
      <td>2</td>
      <td>0.03</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>17.00</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>15.00</td>
      <td>2</td>
      <td>0.01</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>7.75</td>
      <td>4</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>13.00</td>
      <td>2</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 4 columns</p>
</div>




```python
## Create 'discounted' column for groupby 
h2df['discounted'] = h2df['Discount']>0
h2df['discounted'] = h2df['discounted'].map({True:'Discounted',False:'Full Price'})
h2df
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
      <th>Id</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>14.00</td>
      <td>12</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>9.80</td>
      <td>10</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>34.80</td>
      <td>5</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>18.60</td>
      <td>9</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>42.40</td>
      <td>40</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>33.25</td>
      <td>2</td>
      <td>0.03</td>
      <td>Discounted</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>17.00</td>
      <td>1</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>15.00</td>
      <td>2</td>
      <td>0.01</td>
      <td>Discounted</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>7.75</td>
      <td>4</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>13.00</td>
      <td>2</td>
      <td>0.00</td>
      <td>Full Price</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 5 columns</p>
</div>




```python
## create total spent
h2df["Total Spent"] = h2df['UnitPrice'] * h2df['Quantity']
h2df
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
      <th>Id</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
      <th>Total Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>14.00</td>
      <td>12</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>168.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>9.80</td>
      <td>10</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>98.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>34.80</td>
      <td>5</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>174.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>18.60</td>
      <td>9</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>167.4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>42.40</td>
      <td>40</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>1696.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>33.25</td>
      <td>2</td>
      <td>0.03</td>
      <td>Discounted</td>
      <td>66.5</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>17.00</td>
      <td>1</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>15.00</td>
      <td>2</td>
      <td>0.01</td>
      <td>Discounted</td>
      <td>30.0</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>7.75</td>
      <td>4</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>13.00</td>
      <td>2</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 6 columns</p>
</div>




```python
sns.barplot(data=h2df, x='discounted', y='Total Spent', ci=68 )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a25dafc88>




![png](output_80_1.png)


### put data into dict


```python
h2dict = {}
for name in h2df['discounted'].unique():
    h2dict[name] = h2df.groupby('discounted').get_group(name)['Total Spent']
    
h2dict
```




    {'Full Price': 0        168.0
     1         98.0
     2        174.0
     3        167.4
     4       1696.0
              ...  
     2147      14.0
     2148      48.0
     2151      17.0
     2153      31.0
     2154      26.0
     Name: Total Spent, Length: 1317, dtype: float64, 'Discounted': 6       1484.0
     7        252.0
     8        100.8
     9        234.0
     11      2592.0
              ...  
     2144      36.0
     2146      36.0
     2149      68.0
     2150      66.5
     2152      30.0
     Name: Total Spent, Length: 838, dtype: float64}



## STEP 2: Do we meet the assumptions of the chosen test?

Independent t-test (2-sample)

- No significant outliers
- Normality
- Equal Variance

### 0. Check for & Remove Outliers


```python
fig,ax=plt.subplots(figsize=(8,5))
for name,values in h2dict.items():
    sns.distplot(values,label=name, ax=ax)
    
    
ax.legend()
ax.set(title='Total Spent on Full Price vs Discounted Products', ylabel='Density')
```




    [Text(0, 0.5, 'Density'),
     Text(0.5, 1.0, 'Total Spent on Full Price vs Discounted Products')]




![png](output_86_1.png)



```python
for name,values in h2dict.items():
    idx_outs = find_outliers_Z(values)
    print(f"Found {idx_outs.sum()} outliers using Z-score method for {name}.")
    h2dict[name] = values[~idx_outs]
    
```

    Found 19 outliers using Z-score method for Full Price.
    Found 13 outliers using Z-score method for Discounted.


### Check result of outlier removal on plot


```python
fig,ax=plt.subplots(figsize=(8,5))
for name,values in h2dict.items():
    sns.distplot(values,label=name, ax=ax)
    
    
ax.legend()
ax.set(title='Total Spent on Full Price vs Discounted Products', ylabel='Density')
```




    [Text(0, 0.5, 'Density'),
     Text(0.5, 1.0, 'Total Spent on Full Price vs Discounted Products')]




![png](output_89_1.png)


### 1. Test Assumption of Normality


```python
import scipy.stats as stats
```


```python
for key,values in h2dict.items():
    stat,h2p = stats.normaltest(values)
    print(f"Group {key} Normaltest p-value={round(h2p,4)}")
    sig = 'is NOT' if h2p<.05 else 'IS'

    print(f"\t-The data {sig} normal.")
```

    Group Full Price Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Discounted Normaltest p-value=0.0
    	-The data is NOT normal.


### 1B. We don't have normal data:


```python
len(h2dict['Full Price']), len(h2dict['Discounted'])
```




    (1298, 825)



But group sizes (n) are bigger than required 15 for each group so we can safely ignore normality assumption. <br>
So we move on to equal variance assumption.

### 2. Test for Equal Variance


```python
# put data into lists for levene's test args
h2data = []
for key,vals in h2dict.items():
    h2data.append(vals)
```


```python
h2data[1]
```




    6       1484.0
    7        252.0
    8        100.8
    9        234.0
    11      2592.0
             ...  
    2144      36.0
    2146      36.0
    2149      68.0
    2150      66.5
    2152      30.0
    Name: Total Spent, Length: 825, dtype: float64




```python
stat,h2p = stats.levene(*h2data)
print(f"Levene's Test for Equal Variance p-value={round(h2p,4)}")
sig = 'do NOT' if h2p<.05 else 'DO'

print(f"\t-The groups {sig} have equal variance.")
```

    Levene's Test for Equal Variance p-value=0.0
    	-The groups do NOT have equal variance.


### Failed the assumption of equal variance:

Since we don't have equal variance and data is not normal, we need to use a non-parametric version of a t-test. 
 - The non-parametric version for 2 sample is Mann-Whitney U test.
 - Works from medians instead of means. 
 - scipy.stats.mannwhitneyu()

## STEP 3: Interpret Result & Post-Hoc Tests

### Perform hypothesis test from summary table above to get your p-value.

- If p value is < $\alpha$:

    - Reject the null hypothesis.
    - Calculate effect size (e.g. Cohen's $d$)
- If p<.05 AND you have multiple groups (i.e. ANOVA)

    - Must run a pairwise Tukey's test to know which groups were significantly different.
    - Tukey pairwise comparison test
    - statsmodels.stats.multicomp.pairwise_tukeyhsd
- Report statistical power (optional)


```python
stat,h2p = stats.mannwhitneyu(h2data[0],h2data[1])
print(f"Mann Whitney U p-value={round(h2p,4)}")
```

    Mann Whitney U p-value=0.0


Mann Whitney p val defaults to one-sided test, so double that p val for our two sided test, and we will reject null hypothesis. 

### Calculate effect size (e.g. Cohen's  𝑑 )


```python
h2effectsize = Cohen_d(h2dict['Full Price'],h2dict['Discounted'])
h2effectsize
```




    -0.22821293413002144



0.2 effect size is small but still holds up. 

### Report statistical power (optional)

statsmodels.stats.power:
TTestIndPower , TTestPower


```python
len(h2dict['Full Price']+h2dict['Discounted'])
```




    2123




```python
# show the power to see how well we can reject null.
power_analysis.solve_power(effect_size=h2effectsize, nobs1=2123, alpha=h2p)
```




    0.9882912139030121




```python
# power is .99 so yeah we have a solid bet of rejecting null hypothesis and not getting any type 1 errors.
```

## What levels of discount?

> Does discount amount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount? 

### Make dict into dataframes for tukeys


```python
h2df
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
      <th>Id</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
      <th>Total Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>14.00</td>
      <td>12</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>168.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>9.80</td>
      <td>10</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>98.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>34.80</td>
      <td>5</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>174.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>18.60</td>
      <td>9</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>167.4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>42.40</td>
      <td>40</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>1696.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>33.25</td>
      <td>2</td>
      <td>0.03</td>
      <td>Discounted</td>
      <td>66.5</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>17.00</td>
      <td>1</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>15.00</td>
      <td>2</td>
      <td>0.01</td>
      <td>Discounted</td>
      <td>30.0</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>7.75</td>
      <td>4</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>13.00</td>
      <td>2</td>
      <td>0.00</td>
      <td>Full Price</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 6 columns</p>
</div>




```python
# -- Side note, this tukeys test is using original df containing all outliers. 
h2tukey = pairwise_tukeyhsd(h2df['Total Spent'],h2df['Discount'])
h2tukey.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD, FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>   <th>p-adj</th>    <th>lower</th>     <th>upper</th>   <th>reject</th>
</tr>
<tr>
    <td>0.0</td>   <td>0.01</td>  <td>-540.0065</td>   <td>0.9</td>   <td>-3870.951</td> <td>2790.9379</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.02</td>  <td>-539.5065</td>   <td>0.9</td>  <td>-2895.7333</td> <td>1816.7202</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.03</td>  <td>-528.4565</td>   <td>0.9</td>  <td>-2453.0368</td> <td>1396.1237</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.04</td>  <td>-489.0065</td>   <td>0.9</td>   <td>-3819.951</td> <td>2841.9379</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.05</td>  <td>269.9216</td>  <td>0.0362</td>   <td>8.4896</td>   <td>531.3536</td>   <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.06</td>  <td>-502.0065</td>   <td>0.9</td>   <td>-3832.951</td> <td>2828.9379</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.1</td>   <td>17.6565</td>    <td>0.9</td>   <td>-251.6084</td> <td>286.9214</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.15</td>   <td>85.716</td>     <td>0.9</td>   <td>-195.4149</td> <td>366.8469</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.2</td>  <td>122.3933</td>    <td>0.9</td>   <td>-155.5997</td> <td>400.3864</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.25</td>  <td>286.6044</td>   <td>0.045</td>   <td>3.0375</td>   <td>570.1714</td>   <td>True</td> 
</tr>
<tr>
   <td>0.01</td>   <td>0.02</td>     <td>0.5</td>      <td>0.9</td>  <td>-4077.5092</td> <td>4078.5092</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.03</td>    <td>11.55</td>     <td>0.9</td>  <td>-3833.2339</td> <td>3856.3339</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.04</td>    <td>51.0</td>      <td>0.9</td>  <td>-4657.8794</td> <td>4759.8794</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.05</td>  <td>809.9282</td>    <td>0.9</td>  <td>-2528.7394</td> <td>4148.5957</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.06</td>    <td>38.0</td>      <td>0.9</td>  <td>-4670.8794</td> <td>4746.8794</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.1</td>  <td>557.6631</td>    <td>0.9</td>   <td>-2781.627</td> <td>3896.9531</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.15</td>  <td>625.7225</td>    <td>0.9</td>  <td>-2714.5453</td> <td>3965.9904</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.2</td>  <td>662.3999</td>    <td>0.9</td>  <td>-2677.6053</td> <td>4002.4051</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.25</td>   <td>826.611</td>    <td>0.9</td>  <td>-2513.8627</td> <td>4167.0847</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.03</td>    <td>11.05</td>     <td>0.9</td>  <td>-3028.5186</td> <td>3050.6186</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.04</td>    <td>50.5</td>      <td>0.9</td>  <td>-4027.5092</td> <td>4128.5092</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.05</td>  <td>809.4282</td>    <td>0.9</td>   <td>-1557.704</td> <td>3176.5604</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.06</td>    <td>37.5</td>      <td>0.9</td>  <td>-4040.5092</td> <td>4115.5092</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.1</td>  <td>557.1631</td>    <td>0.9</td>   <td>-1810.847</td> <td>2925.1731</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.15</td>  <td>625.2225</td>    <td>0.9</td>  <td>-1744.1661</td> <td>2994.6112</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.2</td>  <td>661.8999</td>    <td>0.9</td>  <td>-1707.1185</td> <td>3030.9183</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.25</td>   <td>826.111</td>    <td>0.9</td>   <td>-1543.568</td> <td>3195.7899</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.04</td>    <td>39.45</td>     <td>0.9</td>  <td>-3805.3339</td> <td>3884.2339</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.05</td>  <td>798.3782</td>    <td>0.9</td>  <td>-1139.5381</td> <td>2736.2944</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.06</td>    <td>26.45</td>     <td>0.9</td>  <td>-3818.3339</td> <td>3871.2339</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.1</td>  <td>546.1131</td>    <td>0.9</td>  <td>-1392.8754</td> <td>2485.1015</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.15</td>  <td>614.1725</td>    <td>0.9</td>  <td>-1326.4993</td> <td>2554.8444</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.2</td>  <td>650.8499</td>    <td>0.9</td>  <td>-1289.3699</td> <td>2591.0697</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.25</td>   <td>815.061</td>    <td>0.9</td>  <td>-1125.9653</td> <td>2756.0872</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.05</td>  <td>758.9282</td>    <td>0.9</td>  <td>-2579.7394</td> <td>4097.5957</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.06</td>    <td>-13.0</td>     <td>0.9</td>  <td>-4721.8794</td> <td>4695.8794</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.1</td>  <td>506.6631</td>    <td>0.9</td>   <td>-2832.627</td> <td>3845.9531</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.15</td>  <td>574.7225</td>    <td>0.9</td>  <td>-2765.5453</td> <td>3914.9904</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.2</td>  <td>611.3999</td>    <td>0.9</td>  <td>-2728.6053</td> <td>3951.4051</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.25</td>   <td>775.611</td>    <td>0.9</td>  <td>-2564.8627</td> <td>4116.0847</td>  <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.06</td>  <td>-771.9282</td>   <td>0.9</td>  <td>-4110.5957</td> <td>2566.7394</td>  <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.1</td>  <td>-252.2651</td> <td>0.4322</td>  <td>-604.4212</td>  <td>99.891</td>    <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.15</td>  <td>-184.2056</td> <td>0.8503</td>  <td>-545.5156</td> <td>177.1043</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.2</td>  <td>-147.5283</td>   <td>0.9</td>   <td>-506.4021</td> <td>211.3456</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.25</td>   <td>16.6828</td>    <td>0.9</td>   <td>-346.5258</td> <td>379.8915</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.1</td>  <td>519.6631</td>    <td>0.9</td>   <td>-2819.627</td> <td>3858.9531</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.15</td>  <td>587.7225</td>    <td>0.9</td>  <td>-2752.5453</td> <td>3927.9904</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.2</td>  <td>624.3999</td>    <td>0.9</td>  <td>-2715.6053</td> <td>3964.4051</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.25</td>   <td>788.611</td>    <td>0.9</td>  <td>-2551.8627</td> <td>4129.0847</td>  <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.15</td>   <td>68.0595</td>    <td>0.9</td>   <td>-298.9579</td> <td>435.0769</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>    <td>0.2</td>  <td>104.7368</td>    <td>0.9</td>   <td>-259.8826</td> <td>469.3562</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.25</td>  <td>268.9479</td>  <td>0.4037</td>  <td>-99.9388</td>  <td>637.8346</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>    <td>0.2</td>   <td>36.6773</td>    <td>0.9</td>   <td>-336.7906</td> <td>410.1453</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>   <td>0.25</td>  <td>200.8884</td>   <td>0.806</td>  <td>-176.7469</td> <td>578.5237</td>   <td>False</td>
</tr>
<tr>
    <td>0.2</td>   <td>0.25</td>  <td>164.2111</td>    <td>0.9</td>   <td>-211.0941</td> <td>539.5163</td>   <td>False</td>
</tr>
</table>




```python
h2_tukeydf = pd.DataFrame(data=h2tukey._results_table.data[1:], columns=h2tukey._results_table.data[0])
h2_tukeydf
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
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>-540.0065</td>
      <td>0.9000</td>
      <td>-3870.9510</td>
      <td>2790.9379</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>-539.5065</td>
      <td>0.9000</td>
      <td>-2895.7333</td>
      <td>1816.7202</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>-528.4565</td>
      <td>0.9000</td>
      <td>-2453.0368</td>
      <td>1396.1237</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>-489.0065</td>
      <td>0.9000</td>
      <td>-3819.9510</td>
      <td>2841.9379</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>269.9216</td>
      <td>0.0362</td>
      <td>8.4896</td>
      <td>531.3536</td>
      <td>True</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>-502.0065</td>
      <td>0.9000</td>
      <td>-3832.9510</td>
      <td>2828.9379</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.00</td>
      <td>0.10</td>
      <td>17.6565</td>
      <td>0.9000</td>
      <td>-251.6084</td>
      <td>286.9214</td>
      <td>False</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.00</td>
      <td>0.15</td>
      <td>85.7160</td>
      <td>0.9000</td>
      <td>-195.4149</td>
      <td>366.8469</td>
      <td>False</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.00</td>
      <td>0.20</td>
      <td>122.3933</td>
      <td>0.9000</td>
      <td>-155.5997</td>
      <td>400.3864</td>
      <td>False</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>286.6044</td>
      <td>0.0450</td>
      <td>3.0375</td>
      <td>570.1714</td>
      <td>True</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.5000</td>
      <td>0.9000</td>
      <td>-4077.5092</td>
      <td>4078.5092</td>
      <td>False</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.01</td>
      <td>0.03</td>
      <td>11.5500</td>
      <td>0.9000</td>
      <td>-3833.2339</td>
      <td>3856.3339</td>
      <td>False</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>51.0000</td>
      <td>0.9000</td>
      <td>-4657.8794</td>
      <td>4759.8794</td>
      <td>False</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>809.9282</td>
      <td>0.9000</td>
      <td>-2528.7394</td>
      <td>4148.5957</td>
      <td>False</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.01</td>
      <td>0.06</td>
      <td>38.0000</td>
      <td>0.9000</td>
      <td>-4670.8794</td>
      <td>4746.8794</td>
      <td>False</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.01</td>
      <td>0.10</td>
      <td>557.6631</td>
      <td>0.9000</td>
      <td>-2781.6270</td>
      <td>3896.9531</td>
      <td>False</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.01</td>
      <td>0.15</td>
      <td>625.7225</td>
      <td>0.9000</td>
      <td>-2714.5453</td>
      <td>3965.9904</td>
      <td>False</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.01</td>
      <td>0.20</td>
      <td>662.3999</td>
      <td>0.9000</td>
      <td>-2677.6053</td>
      <td>4002.4051</td>
      <td>False</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.01</td>
      <td>0.25</td>
      <td>826.6110</td>
      <td>0.9000</td>
      <td>-2513.8627</td>
      <td>4167.0847</td>
      <td>False</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>11.0500</td>
      <td>0.9000</td>
      <td>-3028.5186</td>
      <td>3050.6186</td>
      <td>False</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>50.5000</td>
      <td>0.9000</td>
      <td>-4027.5092</td>
      <td>4128.5092</td>
      <td>False</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>809.4282</td>
      <td>0.9000</td>
      <td>-1557.7040</td>
      <td>3176.5604</td>
      <td>False</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.02</td>
      <td>0.06</td>
      <td>37.5000</td>
      <td>0.9000</td>
      <td>-4040.5092</td>
      <td>4115.5092</td>
      <td>False</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.02</td>
      <td>0.10</td>
      <td>557.1631</td>
      <td>0.9000</td>
      <td>-1810.8470</td>
      <td>2925.1731</td>
      <td>False</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.02</td>
      <td>0.15</td>
      <td>625.2225</td>
      <td>0.9000</td>
      <td>-1744.1661</td>
      <td>2994.6112</td>
      <td>False</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.02</td>
      <td>0.20</td>
      <td>661.8999</td>
      <td>0.9000</td>
      <td>-1707.1185</td>
      <td>3030.9183</td>
      <td>False</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.02</td>
      <td>0.25</td>
      <td>826.1110</td>
      <td>0.9000</td>
      <td>-1543.5680</td>
      <td>3195.7899</td>
      <td>False</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>39.4500</td>
      <td>0.9000</td>
      <td>-3805.3339</td>
      <td>3884.2339</td>
      <td>False</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>798.3782</td>
      <td>0.9000</td>
      <td>-1139.5381</td>
      <td>2736.2944</td>
      <td>False</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.03</td>
      <td>0.06</td>
      <td>26.4500</td>
      <td>0.9000</td>
      <td>-3818.3339</td>
      <td>3871.2339</td>
      <td>False</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.03</td>
      <td>0.10</td>
      <td>546.1131</td>
      <td>0.9000</td>
      <td>-1392.8754</td>
      <td>2485.1015</td>
      <td>False</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.03</td>
      <td>0.15</td>
      <td>614.1725</td>
      <td>0.9000</td>
      <td>-1326.4993</td>
      <td>2554.8444</td>
      <td>False</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.03</td>
      <td>0.20</td>
      <td>650.8499</td>
      <td>0.9000</td>
      <td>-1289.3699</td>
      <td>2591.0697</td>
      <td>False</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.03</td>
      <td>0.25</td>
      <td>815.0610</td>
      <td>0.9000</td>
      <td>-1125.9653</td>
      <td>2756.0872</td>
      <td>False</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>758.9282</td>
      <td>0.9000</td>
      <td>-2579.7394</td>
      <td>4097.5957</td>
      <td>False</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.04</td>
      <td>0.06</td>
      <td>-13.0000</td>
      <td>0.9000</td>
      <td>-4721.8794</td>
      <td>4695.8794</td>
      <td>False</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.04</td>
      <td>0.10</td>
      <td>506.6631</td>
      <td>0.9000</td>
      <td>-2832.6270</td>
      <td>3845.9531</td>
      <td>False</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.04</td>
      <td>0.15</td>
      <td>574.7225</td>
      <td>0.9000</td>
      <td>-2765.5453</td>
      <td>3914.9904</td>
      <td>False</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.04</td>
      <td>0.20</td>
      <td>611.3999</td>
      <td>0.9000</td>
      <td>-2728.6053</td>
      <td>3951.4051</td>
      <td>False</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.04</td>
      <td>0.25</td>
      <td>775.6110</td>
      <td>0.9000</td>
      <td>-2564.8627</td>
      <td>4116.0847</td>
      <td>False</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.05</td>
      <td>0.06</td>
      <td>-771.9282</td>
      <td>0.9000</td>
      <td>-4110.5957</td>
      <td>2566.7394</td>
      <td>False</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0.05</td>
      <td>0.10</td>
      <td>-252.2651</td>
      <td>0.4322</td>
      <td>-604.4212</td>
      <td>99.8910</td>
      <td>False</td>
    </tr>
    <tr>
      <td>42</td>
      <td>0.05</td>
      <td>0.15</td>
      <td>-184.2056</td>
      <td>0.8503</td>
      <td>-545.5156</td>
      <td>177.1043</td>
      <td>False</td>
    </tr>
    <tr>
      <td>43</td>
      <td>0.05</td>
      <td>0.20</td>
      <td>-147.5283</td>
      <td>0.9000</td>
      <td>-506.4021</td>
      <td>211.3456</td>
      <td>False</td>
    </tr>
    <tr>
      <td>44</td>
      <td>0.05</td>
      <td>0.25</td>
      <td>16.6828</td>
      <td>0.9000</td>
      <td>-346.5258</td>
      <td>379.8915</td>
      <td>False</td>
    </tr>
    <tr>
      <td>45</td>
      <td>0.06</td>
      <td>0.10</td>
      <td>519.6631</td>
      <td>0.9000</td>
      <td>-2819.6270</td>
      <td>3858.9531</td>
      <td>False</td>
    </tr>
    <tr>
      <td>46</td>
      <td>0.06</td>
      <td>0.15</td>
      <td>587.7225</td>
      <td>0.9000</td>
      <td>-2752.5453</td>
      <td>3927.9904</td>
      <td>False</td>
    </tr>
    <tr>
      <td>47</td>
      <td>0.06</td>
      <td>0.20</td>
      <td>624.3999</td>
      <td>0.9000</td>
      <td>-2715.6053</td>
      <td>3964.4051</td>
      <td>False</td>
    </tr>
    <tr>
      <td>48</td>
      <td>0.06</td>
      <td>0.25</td>
      <td>788.6110</td>
      <td>0.9000</td>
      <td>-2551.8627</td>
      <td>4129.0847</td>
      <td>False</td>
    </tr>
    <tr>
      <td>49</td>
      <td>0.10</td>
      <td>0.15</td>
      <td>68.0595</td>
      <td>0.9000</td>
      <td>-298.9579</td>
      <td>435.0769</td>
      <td>False</td>
    </tr>
    <tr>
      <td>50</td>
      <td>0.10</td>
      <td>0.20</td>
      <td>104.7368</td>
      <td>0.9000</td>
      <td>-259.8826</td>
      <td>469.3562</td>
      <td>False</td>
    </tr>
    <tr>
      <td>51</td>
      <td>0.10</td>
      <td>0.25</td>
      <td>268.9479</td>
      <td>0.4037</td>
      <td>-99.9388</td>
      <td>637.8346</td>
      <td>False</td>
    </tr>
    <tr>
      <td>52</td>
      <td>0.15</td>
      <td>0.20</td>
      <td>36.6773</td>
      <td>0.9000</td>
      <td>-336.7906</td>
      <td>410.1453</td>
      <td>False</td>
    </tr>
    <tr>
      <td>53</td>
      <td>0.15</td>
      <td>0.25</td>
      <td>200.8884</td>
      <td>0.8060</td>
      <td>-176.7469</td>
      <td>578.5237</td>
      <td>False</td>
    </tr>
    <tr>
      <td>54</td>
      <td>0.20</td>
      <td>0.25</td>
      <td>164.2111</td>
      <td>0.9000</td>
      <td>-211.0941</td>
      <td>539.5163</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
h2_tukeydf.loc[h1_tukeydf['reject']==True]
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
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>269.9216</td>
      <td>0.0362</td>
      <td>8.4896</td>
      <td>531.3536</td>
      <td>True</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.0</td>
      <td>0.15</td>
      <td>85.7160</td>
      <td>0.9000</td>
      <td>-195.4149</td>
      <td>366.8469</td>
      <td>False</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.0</td>
      <td>0.20</td>
      <td>122.3933</td>
      <td>0.9000</td>
      <td>-155.5997</td>
      <td>400.3864</td>
      <td>False</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>286.6044</td>
      <td>0.0450</td>
      <td>3.0375</td>
      <td>570.1714</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



So it looks like the 5%, 15%, 20% and 25% discounts were all significant in effecting the average total spent per order. 


```python
# contains all outliers prior to outlier removal

sns.barplot(data=h2df,x='Discount',y='Total Spent',ci=68)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a263555c0>




![png](output_122_1.png)


# Hypothesis 3

## Question

> Does supplier region have a statistically significant effect on the average total spent per sale? If so, which region(s)?

## Null and Alternative Hypothesis

- $H_0$: Supplier region does not have a significant effect on average total spent per sale. 

- $H_1$: Supplier region does have a significant effect on average total spent per sale.

## STEP 1: Determine the category/type of test based on your data

Type of data: Numerical <br>
    How many groups compared: There are 11 different regions/groups to compare.
    - So we'll use ANOVA & Tukey tests since there are more than two groups. 

### Put data in df


```python
cur.execute("""SELECT *
                FROM Product
                """)
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id',
     'ProductName',
     'SupplierId',
     'CategoryId',
     'QuantityPerUnit',
     'UnitPrice',
     'UnitsInStock',
     'UnitsOnOrder',
     'ReorderLevel',
     'Discontinued']




```python
cur.execute("""SELECT *
                FROM OrderDetail
                """)
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id', 'OrderId', 'ProductId', 'UnitPrice', 'Quantity', 'Discount']




```python
cur.execute("""SELECT *
                FROM Supplier
                """)
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id',
     'CompanyName',
     'ContactName',
     'ContactTitle',
     'Address',
     'City',
     'Region',
     'PostalCode',
     'Country',
     'Phone',
     'Fax',
     'HomePage']




```python
cur.execute("""SELECT od.ID, od.UnitPrice, od.Quantity, s.Region
                FROM OrderDetail od
                JOIN Product p ON od.ProductID = p.ID
                JOIN Supplier s ON p.SupplierID = s.ID
                
            """)
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id', 'UnitPrice', 'Quantity', 'Region']




```python
h3df = pd.DataFrame(cur.fetchall(), columns=col_names)
h3df
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
      <th>Id</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>14.00</td>
      <td>12</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>9.80</td>
      <td>10</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>34.80</td>
      <td>5</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>18.60</td>
      <td>9</td>
      <td>Eastern Asia</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>42.40</td>
      <td>40</td>
      <td>NSW</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>33.25</td>
      <td>2</td>
      <td>Western Europe</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>17.00</td>
      <td>1</td>
      <td>North America</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>15.00</td>
      <td>2</td>
      <td>Northern Europe</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>7.75</td>
      <td>4</td>
      <td>Western Europe</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>13.00</td>
      <td>2</td>
      <td>Western Europe</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 4 columns</p>
</div>




```python
h3df['Region'].unique()
```




    array(['Southern Europe', 'South-East Asia', 'Eastern Asia', 'NSW',
           'North America', 'Northern Europe', 'British Isles', 'Scandinavia',
           'Western Europe', 'South America', 'Victoria'], dtype=object)




```python
## create total spent
h3df["Total Spent"] = h3df['UnitPrice'] * h3df['Quantity']
h3df
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
      <th>Id</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Region</th>
      <th>Total Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>14.00</td>
      <td>12</td>
      <td>Southern Europe</td>
      <td>168.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>9.80</td>
      <td>10</td>
      <td>South-East Asia</td>
      <td>98.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>34.80</td>
      <td>5</td>
      <td>Southern Europe</td>
      <td>174.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>18.60</td>
      <td>9</td>
      <td>Eastern Asia</td>
      <td>167.4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>42.40</td>
      <td>40</td>
      <td>NSW</td>
      <td>1696.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>33.25</td>
      <td>2</td>
      <td>Western Europe</td>
      <td>66.5</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>17.00</td>
      <td>1</td>
      <td>North America</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>15.00</td>
      <td>2</td>
      <td>Northern Europe</td>
      <td>30.0</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>7.75</td>
      <td>4</td>
      <td>Western Europe</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>13.00</td>
      <td>2</td>
      <td>Western Europe</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 5 columns</p>
</div>




```python
ax = sns.barplot(data=h3df, x='Region', y='Total Spent', ci=68)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
```




    [Text(0, 0, 'Southern Europe'),
     Text(0, 0, 'South-East Asia'),
     Text(0, 0, 'Eastern Asia'),
     Text(0, 0, 'NSW'),
     Text(0, 0, 'North America'),
     Text(0, 0, 'Northern Europe'),
     Text(0, 0, 'British Isles'),
     Text(0, 0, 'Scandinavia'),
     Text(0, 0, 'Western Europe'),
     Text(0, 0, 'South America'),
     Text(0, 0, 'Victoria')]




![png](output_138_1.png)


### put data into dict


```python
h3dict = {}
for name in h3df['Region'].unique():
    h3dict[name] = h3df.groupby('Region').get_group(name)['Total Spent']
    
h3dict
```




    {'Southern Europe': 0       168.0
     2       174.0
     9       234.0
     14      200.0
     31      153.6
             ...  
     2077    390.0
     2114    250.0
     2121    210.0
     2137     76.0
     2143     32.0
     Name: Total Spent, Length: 229, dtype: float64,
     'South-East Asia': 1         98.00
     60       920.00
     78       248.00
     96       325.50
     116      372.00
              ...   
     2032     233.40
     2036     420.00
     2051     291.75
     2067    1380.00
     2108    1656.00
     Name: Total Spent, Length: 82, dtype: float64,
     'Eastern Asia': 3       167.40
     19      168.00
     46      288.00
     65      595.20
     74      372.00
              ...  
     2116     60.00
     2128    465.00
     2136     31.00
     2138     24.00
     2139     23.25
     Name: Total Spent, Length: 119, dtype: float64,
     'NSW': 4       1696.0
     6       1484.0
     24       393.0
     101      943.2
     117       84.8
              ...  
     2056    1272.0
     2076    1060.0
     2091     328.0
     2098     820.0
     2147      14.0
     Name: Total Spent, Length: 98, dtype: float64,
     'North America': 5         77.00
     7        252.00
     10       336.00
     18       403.20
     30      1105.00
              ...   
     2134      30.00
     2135      80.00
     2145      28.95
     2148      48.00
     2151      17.00
     Name: Total Spent, Length: 418, dtype: float64,
     'Northern Europe': 8       100.8
     22      380.0
     33       20.8
     59      456.0
     81      300.0
             ...  
     2026    665.0
     2125    360.0
     2142     18.0
     2146     36.0
     2152     30.0
     Name: Total Spent, Length: 153, dtype: float64,
     'British Isles': 11      2592.0
     20       304.0
     29       760.0
     32        80.0
     38       160.0
              ...  
     2124     190.0
     2129      92.0
     2130     456.0
     2131      40.0
     2141      81.0
     Name: Total Spent, Length: 220, dtype: float64,
     'Scandinavia': 12       50.0
     16      640.0
     54      216.0
     57      120.0
     61       48.0
             ...  
     2055     37.5
     2065    900.0
     2072     25.0
     2119    357.5
     2126     36.0
     Name: Total Spent, Length: 175, dtype: float64,
     'Western Europe': 13      1088.0
     15       604.8
     23      1320.0
     25       124.8
     26       877.5
              ...  
     2144      36.0
     2149      68.0
     2150      66.5
     2153      31.0
     2154      26.0
     Name: Total Spent, Length: 447, dtype: float64,
     'South America': 17       54.0
     44      100.8
     72       43.2
     83       43.2
     87       21.6
     120      36.0
     278      36.0
     284      90.0
     292      36.0
     365      54.0
     367      54.0
     525      72.0
     577      90.0
     584     100.8
     596     288.0
     659      63.0
     689      45.0
     712      22.5
     718      36.0
     784     157.5
     825      81.0
     849     112.5
     925     157.5
     981      22.5
     992      13.5
     1054     67.5
     1176     90.0
     1202    157.5
     1361     90.0
     1441     90.0
     1604    180.0
     1623     27.0
     1663     36.0
     1668     54.0
     1685     45.0
     1692    495.0
     1749    157.5
     1811    112.5
     1844     45.0
     1864    135.0
     1888    360.0
     1899     90.0
     1933     54.0
     1941    135.0
     1966     54.0
     1970     45.0
     2011    135.0
     2023     94.5
     2066     45.0
     2074     67.5
     2122     90.0
     Name: Total Spent, dtype: float64,
     'Victoria': 21       486.50
     37       252.00
     43       834.00
     49       936.00
     50       240.00
              ...   
     2096    3003.00
     2103      52.35
     2113     523.50
     2123     244.30
     2140      34.90
     Name: Total Spent, Length: 163, dtype: float64}



## STEP 2: Do we meet the assumptions of the chosen test?

ANOVA tukey

- No significant outliers
- Normality
- Equal Variance

### 0. Check for & Remove Outliers


```python
fig,ax=plt.subplots(figsize=(8,5))
for name,values in h3dict.items():
    sns.distplot(values,label=name, ax=ax)
    
    
ax.legend()
ax.set(title='Total Spent by Region', ylabel='Density')
```




    [Text(0, 0.5, 'Density'), Text(0.5, 1.0, 'Total Spent by Region')]




![png](output_144_1.png)


Something is seriously skewing the sales data for these regions. Looks like Western Europe and South America. 


```python
h3df.describe()
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
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Total Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2155.000000</td>
      <td>2155.000000</td>
      <td>2155.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>26.218520</td>
      <td>23.812993</td>
      <td>628.519067</td>
    </tr>
    <tr>
      <td>std</td>
      <td>29.827418</td>
      <td>19.022047</td>
      <td>1036.466980</td>
    </tr>
    <tr>
      <td>min</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>12.000000</td>
      <td>10.000000</td>
      <td>154.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>18.400000</td>
      <td>20.000000</td>
      <td>360.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>32.000000</td>
      <td>30.000000</td>
      <td>722.250000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>263.500000</td>
      <td>130.000000</td>
      <td>15810.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for name,values in h3dict.items():
    idx_outs = find_outliers_Z(values)
    print(f"Found {idx_outs.sum()} outliers using Z-score method for {name}.")
    h3dict[name] = values[~idx_outs]
    
```

    Found 6 outliers using Z-score method for Southern Europe.
    Found 1 outliers using Z-score method for South-East Asia.
    Found 2 outliers using Z-score method for Eastern Asia.
    Found 2 outliers using Z-score method for NSW.
    Found 11 outliers using Z-score method for North America.
    Found 2 outliers using Z-score method for Northern Europe.
    Found 4 outliers using Z-score method for British Isles.
    Found 2 outliers using Z-score method for Scandinavia.
    Found 13 outliers using Z-score method for Western Europe.
    Found 2 outliers using Z-score method for South America.
    Found 4 outliers using Z-score method for Victoria.


### 1. Test Assumption of Normality


```python
for key,values in h3dict.items():
    stat,h3p = stats.normaltest(values)
    print(f"Group {key} Normaltest p-value={round(h3p,4)}")
    sig = 'is NOT' if h3p<.05 else 'IS'

    print(f"\t-The data {sig} normal.")
```

    Group Southern Europe Normaltest p-value=0.0
    	-The data is NOT normal.
    Group South-East Asia Normaltest p-value=0.0001
    	-The data is NOT normal.
    Group Eastern Asia Normaltest p-value=0.0
    	-The data is NOT normal.
    Group NSW Normaltest p-value=0.0
    	-The data is NOT normal.
    Group North America Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Northern Europe Normaltest p-value=0.0
    	-The data is NOT normal.
    Group British Isles Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Scandinavia Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Western Europe Normaltest p-value=0.0
    	-The data is NOT normal.
    Group South America Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Victoria Normaltest p-value=0.0
    	-The data is NOT normal.


### 1B. We don't have normal data:

So in order to ignore non-normality, if we have 2-9 groups, each group needs n >= 15.
If we have 10-12 groups, we need each group to have n>20. We have 11 groups. 


```python
for name in h3dict.keys():
    region = name
    print(f"{region}: ", len(h3dict[name]))
```

    Southern Europe:  223
    South-East Asia:  81
    Eastern Asia:  117
    NSW:  96
    North America:  407
    Northern Europe:  151
    British Isles:  216
    Scandinavia:  173
    Western Europe:  434
    South America:  49
    Victoria:  159


All 11 group sizes (n) are bigger than required 20 for each group so we can safely ignore normality assumption. <br>
So we move on to equal variance assumption.

### 2. Test for Equal Variance


```python
# put data into lists for levene's test args
h3data = []
for key,vals in h3dict.items():
    h3data.append(vals)
```


```python
h3data[1]
```




    1         98.00
    60       920.00
    78       248.00
    96       325.50
    116      372.00
             ...   
    2032     233.40
    2036     420.00
    2051     291.75
    2067    1380.00
    2108    1656.00
    Name: Total Spent, Length: 81, dtype: float64




```python
stat,h3p = stats.levene(*h3data)
print(f"Levene's Test for Equal Variance p-value={round(h3p,4)}")
sig = 'do NOT' if h3p<.05 else 'DO'

print(f"\t-The groups {sig} have equal variance.")
```

    Levene's Test for Equal Variance p-value=0.0
    	-The groups do NOT have equal variance.


### Failed the assumption of equal variance:

Since we don't have equal variance and data is not normal, we need to use a non-parametric version of a t-test. 
 - The non-parametric version for ANOVA is Kruskal-Wallis.
 - Works from medians instead of means. 
 - scipy.stats.kruskal

## STEP 3: Interpret Result & Post-Hoc Tests

### Perform hypothesis test from summary table above to get your p-value.

- If p value is < $\alpha$:

    - Reject the null hypothesis.
    - Calculate effect size (e.g. Cohen's $d$)
- If p<.05 AND you have multiple groups (i.e. ANOVA)

    - Must run a pairwise Tukey's test to know which groups were significantly different.
    - Tukey pairwise comparison test
    - statsmodels.stats.multicomp.pairwise_tukeyhsd
- Report statistical power (optional)


```python
stat,h3p = stats.kruskal(*h3data)
print(f"Kruskal-Wallis p-value={round(h3p,4)}")
```

    Kruskal-Wallis p-value=0.0


Kruskal-Wallis p val shows that the null hypothesis should be rejected but need to check with a tukeys test to see which groups specifically are significant. 

### Checking Tukeys for specific group significance.

### If p<.05 AND you have multiple groups (i.e. ANOVA)

- Must run a pairwise Tukey's test to know which groups were significantly different.
- Tukey pairwise comparison test
- statsmodels.stats.multicomp.pairwise_tukeyhsd

> Does supplier region have a statistically significant effect on the total sales of the supplier? If so, which region(s)?


```python
h3df
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
      <th>Id</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Region</th>
      <th>Total Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>14.00</td>
      <td>12</td>
      <td>Southern Europe</td>
      <td>168.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>9.80</td>
      <td>10</td>
      <td>South-East Asia</td>
      <td>98.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>34.80</td>
      <td>5</td>
      <td>Southern Europe</td>
      <td>174.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>18.60</td>
      <td>9</td>
      <td>Eastern Asia</td>
      <td>167.4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>42.40</td>
      <td>40</td>
      <td>NSW</td>
      <td>1696.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077/64</td>
      <td>33.25</td>
      <td>2</td>
      <td>Western Europe</td>
      <td>66.5</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077/66</td>
      <td>17.00</td>
      <td>1</td>
      <td>North America</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077/73</td>
      <td>15.00</td>
      <td>2</td>
      <td>Northern Europe</td>
      <td>30.0</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077/75</td>
      <td>7.75</td>
      <td>4</td>
      <td>Western Europe</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077/77</td>
      <td>13.00</td>
      <td>2</td>
      <td>Western Europe</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 5 columns</p>
</div>




```python
def prep_data_for_tukeys(data):
    """Accepts a dictionary with group names as the keys 
    and pandas series as the values. 
    Returns a dataframe ready for tukeys test:
    - with a 'data' column and a 'group' column for sms.stats.multicomp.pairwise_tukeyhsd 
    Example Use:
    df_tukey = prep_data_for_tukeys(grp_data)
    tukey = sms.stats.multicomp.pairwise_tukeyhsd(df_tukey['data'], df_tukey['group'])
    tukey.summary()
    """
    df_tukey = pd.DataFrame(columns=['data','group'])
    for k,v in  data.items():
        grp_df = v.rename('data').to_frame() 
        grp_df['group'] = k
        df_tukey=pd.concat([df_tukey, grp_df],axis=0)

    ## New lines added to ensure compatibility with tukey's test
    df_tukey['group'] = df_tukey['group'].astype('str')
    df_tukey['data'] = df_tukey['data'].astype('float')
    return df_tukey
```


```python
h3tukeyprepdf = prep_data_for_tukeys(h3dict)
h3tukeyprepdf
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
      <th>data</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>168.00</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>2</td>
      <td>174.00</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>9</td>
      <td>234.00</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>14</td>
      <td>200.00</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>31</td>
      <td>153.60</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2092</td>
      <td>180.00</td>
      <td>Victoria</td>
    </tr>
    <tr>
      <td>2103</td>
      <td>52.35</td>
      <td>Victoria</td>
    </tr>
    <tr>
      <td>2113</td>
      <td>523.50</td>
      <td>Victoria</td>
    </tr>
    <tr>
      <td>2123</td>
      <td>244.30</td>
      <td>Victoria</td>
    </tr>
    <tr>
      <td>2140</td>
      <td>34.90</td>
      <td>Victoria</td>
    </tr>
  </tbody>
</table>
<p>2106 rows × 2 columns</p>
</div>




```python
h3tukey = pairwise_tukeyhsd(h3tukeyprepdf['data'],h3tukeyprepdf['group'])
h3tukey.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD, FWER=0.05</caption>
<tr>
      <th>group1</th>          <th>group2</th>      <th>meandiff</th>   <th>p-adj</th>   <th>lower</th>     <th>upper</th>   <th>reject</th>
</tr>
<tr>
   <td>British Isles</td>   <td>Eastern Asia</td>    <td>14.655</td>     <td>0.9</td>  <td>-207.9509</td> <td>237.2609</td>   <td>False</td>
</tr>
<tr>
   <td>British Isles</td>        <td>NSW</td>       <td>280.1238</td>  <td>0.0071</td>  <td>42.2486</td>   <td>517.999</td>   <td>True</td> 
</tr>
<tr>
   <td>British Isles</td>   <td>North America</td>  <td>128.9323</td>  <td>0.2793</td> <td>-34.3181</td>  <td>292.1827</td>   <td>False</td>
</tr>
<tr>
   <td>British Isles</td>  <td>Northern Europe</td> <td>-20.4693</td>    <td>0.9</td>  <td>-226.1777</td> <td>185.2391</td>   <td>False</td>
</tr>
<tr>
   <td>British Isles</td>    <td>Scandinavia</td>    <td>81.4215</td>    <td>0.9</td>  <td>-116.4391</td>  <td>279.282</td>   <td>False</td>
</tr>
<tr>
   <td>British Isles</td>   <td>South America</td>  <td>-257.8523</td> <td>0.1968</td> <td>-564.7067</td>  <td>49.0021</td>   <td>False</td>
</tr>
<tr>
   <td>British Isles</td>  <td>South-East Asia</td> <td>182.6815</td>  <td>0.4172</td> <td>-69.9824</td>  <td>435.3454</td>   <td>False</td>
</tr>
<tr>
   <td>British Isles</td>  <td>Southern Europe</td> <td>179.2675</td>   <td>0.068</td>  <td>-5.867</td>    <td>364.402</td>   <td>False</td>
</tr>
<tr>
   <td>British Isles</td>     <td>Victoria</td>     <td>306.9615</td>   <td>0.001</td> <td>104.3218</td>  <td>509.6012</td>   <td>True</td> 
</tr>
<tr>
   <td>British Isles</td>  <td>Western Europe</td>  <td>531.6876</td>   <td>0.001</td> <td>370.2074</td>  <td>693.1678</td>   <td>True</td> 
</tr>
<tr>
   <td>Eastern Asia</td>         <td>NSW</td>       <td>265.4689</td>   <td>0.053</td>  <td>-1.583</td>   <td>532.5207</td>   <td>False</td>
</tr>
<tr>
   <td>Eastern Asia</td>    <td>North America</td>  <td>114.2773</td>  <td>0.7463</td> <td>-89.1503</td>  <td>317.7049</td>   <td>False</td>
</tr>
<tr>
   <td>Eastern Asia</td>   <td>Northern Europe</td> <td>-35.1243</td>    <td>0.9</td>  <td>-273.9716</td>  <td>203.723</td>   <td>False</td>
</tr>
<tr>
   <td>Eastern Asia</td>     <td>Scandinavia</td>    <td>66.7665</td>    <td>0.9</td>  <td>-165.3561</td> <td>298.8891</td>   <td>False</td>
</tr>
<tr>
   <td>Eastern Asia</td>    <td>South America</td>  <td>-272.5073</td> <td>0.2187</td> <td>-602.4947</td>  <td>57.4802</td>   <td>False</td>
</tr>
<tr>
   <td>Eastern Asia</td>   <td>South-East Asia</td> <td>168.0265</td>  <td>0.6709</td> <td>-112.2789</td> <td>448.3319</td>   <td>False</td>
</tr>
<tr>
   <td>Eastern Asia</td>   <td>Southern Europe</td> <td>164.6125</td>  <td>0.3714</td> <td>-56.7624</td>  <td>385.9874</td>   <td>False</td>
</tr>
<tr>
   <td>Eastern Asia</td>      <td>Victoria</td>     <td>292.3065</td>  <td>0.0033</td>  <td>56.097</td>    <td>528.516</td>   <td>True</td> 
</tr>
<tr>
   <td>Eastern Asia</td>   <td>Western Europe</td>  <td>517.0326</td>   <td>0.001</td> <td>315.0229</td>  <td>719.0424</td>   <td>True</td> 
</tr>
<tr>
        <td>NSW</td>        <td>North America</td>  <td>-151.1916</td> <td>0.4953</td> <td>-371.2234</td>  <td>68.8403</td>   <td>False</td>
</tr>
<tr>
        <td>NSW</td>       <td>Northern Europe</td> <td>-300.5932</td> <td>0.0063</td> <td>-553.732</td>  <td>-47.4544</td>   <td>True</td> 
</tr>
<tr>
        <td>NSW</td>         <td>Scandinavia</td>   <td>-198.7023</td> <td>0.2522</td> <td>-445.5061</td>  <td>48.1014</td>   <td>False</td>
</tr>
<tr>
        <td>NSW</td>        <td>South America</td>  <td>-537.9761</td>  <td>0.001</td> <td>-878.4507</td> <td>-197.5016</td>  <td>True</td> 
</tr>
<tr>
        <td>NSW</td>       <td>South-East Asia</td> <td>-97.4424</td>    <td>0.9</td>  <td>-390.0212</td> <td>195.1364</td>   <td>False</td>
</tr>
<tr>
        <td>NSW</td>       <td>Southern Europe</td> <td>-100.8564</td>   <td>0.9</td>   <td>-337.58</td>  <td>135.8673</td>   <td>False</td>
</tr>
<tr>
        <td>NSW</td>          <td>Victoria</td>      <td>26.8376</td>    <td>0.9</td>  <td>-223.8139</td> <td>277.4891</td>   <td>False</td>
</tr>
<tr>
        <td>NSW</td>       <td>Western Europe</td>  <td>251.5638</td>  <td>0.0098</td>  <td>32.8421</td>  <td>470.2854</td>   <td>True</td> 
</tr>
<tr>
   <td>North America</td>  <td>Northern Europe</td> <td>-149.4016</td> <td>0.2464</td> <td>-334.1862</td>  <td>35.383</td>    <td>False</td>
</tr>
<tr>
   <td>North America</td>    <td>Scandinavia</td>   <td>-47.5108</td>    <td>0.9</td>  <td>-223.517</td>  <td>128.4955</td>   <td>False</td>
</tr>
<tr>
   <td>North America</td>   <td>South America</td>  <td>-386.7846</td> <td>0.0011</td> <td>-680.0234</td> <td>-93.5457</td>   <td>True</td> 
</tr>
<tr>
   <td>North America</td>  <td>South-East Asia</td>  <td>53.7492</td>    <td>0.9</td>  <td>-182.1924</td> <td>289.6908</td>   <td>False</td>
</tr>
<tr>
   <td>North America</td>  <td>Southern Europe</td>  <td>50.3352</td>    <td>0.9</td>  <td>-111.2327</td> <td>211.9031</td>   <td>False</td>
</tr>
<tr>
   <td>North America</td>     <td>Victoria</td>     <td>178.0292</td>  <td>0.0599</td>  <td>-3.333</td>   <td>359.3914</td>   <td>False</td>
</tr>
<tr>
   <td>North America</td>  <td>Western Europe</td>  <td>402.7553</td>   <td>0.001</td> <td>268.9448</td>  <td>536.5659</td>   <td>True</td> 
</tr>
<tr>
  <td>Northern Europe</td>   <td>Scandinavia</td>   <td>101.8908</td>    <td>0.9</td>  <td>-114.0801</td> <td>317.8618</td>   <td>False</td>
</tr>
<tr>
  <td>Northern Europe</td>  <td>South America</td>  <td>-237.383</td>  <td>0.3693</td> <td>-556.2157</td>  <td>81.4497</td>   <td>False</td>
</tr>
<tr>
  <td>Northern Europe</td> <td>South-East Asia</td> <td>203.1508</td>  <td>0.3349</td> <td>-63.9329</td>  <td>470.2345</td>   <td>False</td>
</tr>
<tr>
  <td>Northern Europe</td> <td>Southern Europe</td> <td>199.7368</td>  <td>0.0624</td>  <td>-4.6389</td>  <td>404.1125</td>   <td>False</td>
</tr>
<tr>
  <td>Northern Europe</td>    <td>Victoria</td>     <td>327.4308</td>   <td>0.001</td> <td>107.0731</td>  <td>547.7885</td>   <td>True</td> 
</tr>
<tr>
  <td>Northern Europe</td> <td>Western Europe</td>  <td>552.1569</td>   <td>0.001</td> <td>368.9344</td>  <td>735.3795</td>   <td>True</td> 
</tr>
<tr>
    <td>Scandinavia</td>    <td>South America</td>  <td>-339.2738</td> <td>0.0216</td> <td>-653.1004</td> <td>-25.4472</td>   <td>True</td> 
</tr>
<tr>
    <td>Scandinavia</td>   <td>South-East Asia</td>  <td>101.26</td>     <td>0.9</td>  <td>-159.8273</td> <td>362.3472</td>   <td>False</td>
</tr>
<tr>
    <td>Scandinavia</td>   <td>Southern Europe</td>  <td>97.846</td>   <td>0.8739</td> <td>-98.6286</td>  <td>294.3206</td>   <td>False</td>
</tr>
<tr>
    <td>Scandinavia</td>      <td>Victoria</td>      <td>225.54</td>   <td>0.0275</td>  <td>12.4898</td>  <td>438.5901</td>   <td>True</td> 
</tr>
<tr>
    <td>Scandinavia</td>   <td>Western Europe</td>  <td>450.2661</td>   <td>0.001</td> <td>275.9005</td>  <td>624.6317</td>   <td>True</td> 
</tr>
<tr>
   <td>South America</td>  <td>South-East Asia</td> <td>440.5338</td>  <td>0.0027</td>  <td>89.5674</td>  <td>791.5001</td>   <td>True</td> 
</tr>
<tr>
   <td>South America</td>  <td>Southern Europe</td> <td>437.1198</td>   <td>0.001</td> <td>131.1572</td>  <td>743.0824</td>   <td>True</td> 
</tr>
<tr>
   <td>South America</td>     <td>Victoria</td>     <td>564.8138</td>   <td>0.001</td> <td>247.9523</td>  <td>881.6752</td>   <td>True</td> 
</tr>
<tr>
   <td>South America</td>  <td>Western Europe</td>  <td>789.5399</td>   <td>0.001</td> <td>497.2829</td>  <td>1081.797</td>   <td>True</td> 
</tr>
<tr>
  <td>South-East Asia</td> <td>Southern Europe</td>  <td>-3.414</td>     <td>0.9</td>  <td>-254.9941</td>  <td>248.166</td>   <td>False</td>
</tr>
<tr>
  <td>South-East Asia</td>    <td>Victoria</td>      <td>124.28</td>     <td>0.9</td>  <td>-140.4474</td> <td>389.0074</td>   <td>False</td>
</tr>
<tr>
  <td>South-East Asia</td> <td>Western Europe</td>  <td>349.0061</td>   <td>0.001</td> <td>114.2859</td>  <td>583.7264</td>   <td>True</td> 
</tr>
<tr>
  <td>Southern Europe</td>    <td>Victoria</td>      <td>127.694</td>   <td>0.601</td> <td>-73.5927</td>  <td>328.9807</td>   <td>False</td>
</tr>
<tr>
  <td>Southern Europe</td> <td>Western Europe</td>  <td>352.4201</td>   <td>0.001</td> <td>192.6411</td>  <td>512.1991</td>   <td>True</td> 
</tr>
<tr>
     <td>Victoria</td>     <td>Western Europe</td>  <td>224.7262</td>  <td>0.0028</td>  <td>44.9557</td>  <td>404.4966</td>   <td>True</td> 
</tr>
</table>




```python
h3tukeydf = pd.DataFrame(data=h3tukey._results_table.data[1:], columns=h3tukey._results_table.data[0])
h3tukeydf
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
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>British Isles</td>
      <td>Eastern Asia</td>
      <td>14.6550</td>
      <td>0.9000</td>
      <td>-207.9509</td>
      <td>237.2609</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>British Isles</td>
      <td>NSW</td>
      <td>280.1238</td>
      <td>0.0071</td>
      <td>42.2486</td>
      <td>517.9990</td>
      <td>True</td>
    </tr>
    <tr>
      <td>2</td>
      <td>British Isles</td>
      <td>North America</td>
      <td>128.9323</td>
      <td>0.2793</td>
      <td>-34.3181</td>
      <td>292.1827</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>British Isles</td>
      <td>Northern Europe</td>
      <td>-20.4693</td>
      <td>0.9000</td>
      <td>-226.1777</td>
      <td>185.2391</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>British Isles</td>
      <td>Scandinavia</td>
      <td>81.4215</td>
      <td>0.9000</td>
      <td>-116.4391</td>
      <td>279.2820</td>
      <td>False</td>
    </tr>
    <tr>
      <td>5</td>
      <td>British Isles</td>
      <td>South America</td>
      <td>-257.8523</td>
      <td>0.1968</td>
      <td>-564.7067</td>
      <td>49.0021</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6</td>
      <td>British Isles</td>
      <td>South-East Asia</td>
      <td>182.6815</td>
      <td>0.4172</td>
      <td>-69.9824</td>
      <td>435.3454</td>
      <td>False</td>
    </tr>
    <tr>
      <td>7</td>
      <td>British Isles</td>
      <td>Southern Europe</td>
      <td>179.2675</td>
      <td>0.0680</td>
      <td>-5.8670</td>
      <td>364.4020</td>
      <td>False</td>
    </tr>
    <tr>
      <td>8</td>
      <td>British Isles</td>
      <td>Victoria</td>
      <td>306.9615</td>
      <td>0.0010</td>
      <td>104.3218</td>
      <td>509.6012</td>
      <td>True</td>
    </tr>
    <tr>
      <td>9</td>
      <td>British Isles</td>
      <td>Western Europe</td>
      <td>531.6876</td>
      <td>0.0010</td>
      <td>370.2074</td>
      <td>693.1678</td>
      <td>True</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Eastern Asia</td>
      <td>NSW</td>
      <td>265.4689</td>
      <td>0.0530</td>
      <td>-1.5830</td>
      <td>532.5207</td>
      <td>False</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Eastern Asia</td>
      <td>North America</td>
      <td>114.2773</td>
      <td>0.7463</td>
      <td>-89.1503</td>
      <td>317.7049</td>
      <td>False</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Eastern Asia</td>
      <td>Northern Europe</td>
      <td>-35.1243</td>
      <td>0.9000</td>
      <td>-273.9716</td>
      <td>203.7230</td>
      <td>False</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Eastern Asia</td>
      <td>Scandinavia</td>
      <td>66.7665</td>
      <td>0.9000</td>
      <td>-165.3561</td>
      <td>298.8891</td>
      <td>False</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Eastern Asia</td>
      <td>South America</td>
      <td>-272.5073</td>
      <td>0.2187</td>
      <td>-602.4947</td>
      <td>57.4802</td>
      <td>False</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Eastern Asia</td>
      <td>South-East Asia</td>
      <td>168.0265</td>
      <td>0.6709</td>
      <td>-112.2789</td>
      <td>448.3319</td>
      <td>False</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Eastern Asia</td>
      <td>Southern Europe</td>
      <td>164.6125</td>
      <td>0.3714</td>
      <td>-56.7624</td>
      <td>385.9874</td>
      <td>False</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Eastern Asia</td>
      <td>Victoria</td>
      <td>292.3065</td>
      <td>0.0033</td>
      <td>56.0970</td>
      <td>528.5160</td>
      <td>True</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Eastern Asia</td>
      <td>Western Europe</td>
      <td>517.0326</td>
      <td>0.0010</td>
      <td>315.0229</td>
      <td>719.0424</td>
      <td>True</td>
    </tr>
    <tr>
      <td>19</td>
      <td>NSW</td>
      <td>North America</td>
      <td>-151.1916</td>
      <td>0.4953</td>
      <td>-371.2234</td>
      <td>68.8403</td>
      <td>False</td>
    </tr>
    <tr>
      <td>20</td>
      <td>NSW</td>
      <td>Northern Europe</td>
      <td>-300.5932</td>
      <td>0.0063</td>
      <td>-553.7320</td>
      <td>-47.4544</td>
      <td>True</td>
    </tr>
    <tr>
      <td>21</td>
      <td>NSW</td>
      <td>Scandinavia</td>
      <td>-198.7023</td>
      <td>0.2522</td>
      <td>-445.5061</td>
      <td>48.1014</td>
      <td>False</td>
    </tr>
    <tr>
      <td>22</td>
      <td>NSW</td>
      <td>South America</td>
      <td>-537.9761</td>
      <td>0.0010</td>
      <td>-878.4507</td>
      <td>-197.5016</td>
      <td>True</td>
    </tr>
    <tr>
      <td>23</td>
      <td>NSW</td>
      <td>South-East Asia</td>
      <td>-97.4424</td>
      <td>0.9000</td>
      <td>-390.0212</td>
      <td>195.1364</td>
      <td>False</td>
    </tr>
    <tr>
      <td>24</td>
      <td>NSW</td>
      <td>Southern Europe</td>
      <td>-100.8564</td>
      <td>0.9000</td>
      <td>-337.5800</td>
      <td>135.8673</td>
      <td>False</td>
    </tr>
    <tr>
      <td>25</td>
      <td>NSW</td>
      <td>Victoria</td>
      <td>26.8376</td>
      <td>0.9000</td>
      <td>-223.8139</td>
      <td>277.4891</td>
      <td>False</td>
    </tr>
    <tr>
      <td>26</td>
      <td>NSW</td>
      <td>Western Europe</td>
      <td>251.5638</td>
      <td>0.0098</td>
      <td>32.8421</td>
      <td>470.2854</td>
      <td>True</td>
    </tr>
    <tr>
      <td>27</td>
      <td>North America</td>
      <td>Northern Europe</td>
      <td>-149.4016</td>
      <td>0.2464</td>
      <td>-334.1862</td>
      <td>35.3830</td>
      <td>False</td>
    </tr>
    <tr>
      <td>28</td>
      <td>North America</td>
      <td>Scandinavia</td>
      <td>-47.5108</td>
      <td>0.9000</td>
      <td>-223.5170</td>
      <td>128.4955</td>
      <td>False</td>
    </tr>
    <tr>
      <td>29</td>
      <td>North America</td>
      <td>South America</td>
      <td>-386.7846</td>
      <td>0.0011</td>
      <td>-680.0234</td>
      <td>-93.5457</td>
      <td>True</td>
    </tr>
    <tr>
      <td>30</td>
      <td>North America</td>
      <td>South-East Asia</td>
      <td>53.7492</td>
      <td>0.9000</td>
      <td>-182.1924</td>
      <td>289.6908</td>
      <td>False</td>
    </tr>
    <tr>
      <td>31</td>
      <td>North America</td>
      <td>Southern Europe</td>
      <td>50.3352</td>
      <td>0.9000</td>
      <td>-111.2327</td>
      <td>211.9031</td>
      <td>False</td>
    </tr>
    <tr>
      <td>32</td>
      <td>North America</td>
      <td>Victoria</td>
      <td>178.0292</td>
      <td>0.0599</td>
      <td>-3.3330</td>
      <td>359.3914</td>
      <td>False</td>
    </tr>
    <tr>
      <td>33</td>
      <td>North America</td>
      <td>Western Europe</td>
      <td>402.7553</td>
      <td>0.0010</td>
      <td>268.9448</td>
      <td>536.5659</td>
      <td>True</td>
    </tr>
    <tr>
      <td>34</td>
      <td>Northern Europe</td>
      <td>Scandinavia</td>
      <td>101.8908</td>
      <td>0.9000</td>
      <td>-114.0801</td>
      <td>317.8618</td>
      <td>False</td>
    </tr>
    <tr>
      <td>35</td>
      <td>Northern Europe</td>
      <td>South America</td>
      <td>-237.3830</td>
      <td>0.3693</td>
      <td>-556.2157</td>
      <td>81.4497</td>
      <td>False</td>
    </tr>
    <tr>
      <td>36</td>
      <td>Northern Europe</td>
      <td>South-East Asia</td>
      <td>203.1508</td>
      <td>0.3349</td>
      <td>-63.9329</td>
      <td>470.2345</td>
      <td>False</td>
    </tr>
    <tr>
      <td>37</td>
      <td>Northern Europe</td>
      <td>Southern Europe</td>
      <td>199.7368</td>
      <td>0.0624</td>
      <td>-4.6389</td>
      <td>404.1125</td>
      <td>False</td>
    </tr>
    <tr>
      <td>38</td>
      <td>Northern Europe</td>
      <td>Victoria</td>
      <td>327.4308</td>
      <td>0.0010</td>
      <td>107.0731</td>
      <td>547.7885</td>
      <td>True</td>
    </tr>
    <tr>
      <td>39</td>
      <td>Northern Europe</td>
      <td>Western Europe</td>
      <td>552.1569</td>
      <td>0.0010</td>
      <td>368.9344</td>
      <td>735.3795</td>
      <td>True</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Scandinavia</td>
      <td>South America</td>
      <td>-339.2738</td>
      <td>0.0216</td>
      <td>-653.1004</td>
      <td>-25.4472</td>
      <td>True</td>
    </tr>
    <tr>
      <td>41</td>
      <td>Scandinavia</td>
      <td>South-East Asia</td>
      <td>101.2600</td>
      <td>0.9000</td>
      <td>-159.8273</td>
      <td>362.3472</td>
      <td>False</td>
    </tr>
    <tr>
      <td>42</td>
      <td>Scandinavia</td>
      <td>Southern Europe</td>
      <td>97.8460</td>
      <td>0.8739</td>
      <td>-98.6286</td>
      <td>294.3206</td>
      <td>False</td>
    </tr>
    <tr>
      <td>43</td>
      <td>Scandinavia</td>
      <td>Victoria</td>
      <td>225.5400</td>
      <td>0.0275</td>
      <td>12.4898</td>
      <td>438.5901</td>
      <td>True</td>
    </tr>
    <tr>
      <td>44</td>
      <td>Scandinavia</td>
      <td>Western Europe</td>
      <td>450.2661</td>
      <td>0.0010</td>
      <td>275.9005</td>
      <td>624.6317</td>
      <td>True</td>
    </tr>
    <tr>
      <td>45</td>
      <td>South America</td>
      <td>South-East Asia</td>
      <td>440.5338</td>
      <td>0.0027</td>
      <td>89.5674</td>
      <td>791.5001</td>
      <td>True</td>
    </tr>
    <tr>
      <td>46</td>
      <td>South America</td>
      <td>Southern Europe</td>
      <td>437.1198</td>
      <td>0.0010</td>
      <td>131.1572</td>
      <td>743.0824</td>
      <td>True</td>
    </tr>
    <tr>
      <td>47</td>
      <td>South America</td>
      <td>Victoria</td>
      <td>564.8138</td>
      <td>0.0010</td>
      <td>247.9523</td>
      <td>881.6752</td>
      <td>True</td>
    </tr>
    <tr>
      <td>48</td>
      <td>South America</td>
      <td>Western Europe</td>
      <td>789.5399</td>
      <td>0.0010</td>
      <td>497.2829</td>
      <td>1081.7970</td>
      <td>True</td>
    </tr>
    <tr>
      <td>49</td>
      <td>South-East Asia</td>
      <td>Southern Europe</td>
      <td>-3.4140</td>
      <td>0.9000</td>
      <td>-254.9941</td>
      <td>248.1660</td>
      <td>False</td>
    </tr>
    <tr>
      <td>50</td>
      <td>South-East Asia</td>
      <td>Victoria</td>
      <td>124.2800</td>
      <td>0.9000</td>
      <td>-140.4474</td>
      <td>389.0074</td>
      <td>False</td>
    </tr>
    <tr>
      <td>51</td>
      <td>South-East Asia</td>
      <td>Western Europe</td>
      <td>349.0061</td>
      <td>0.0010</td>
      <td>114.2859</td>
      <td>583.7264</td>
      <td>True</td>
    </tr>
    <tr>
      <td>52</td>
      <td>Southern Europe</td>
      <td>Victoria</td>
      <td>127.6940</td>
      <td>0.6010</td>
      <td>-73.5927</td>
      <td>328.9807</td>
      <td>False</td>
    </tr>
    <tr>
      <td>53</td>
      <td>Southern Europe</td>
      <td>Western Europe</td>
      <td>352.4201</td>
      <td>0.0010</td>
      <td>192.6411</td>
      <td>512.1991</td>
      <td>True</td>
    </tr>
    <tr>
      <td>54</td>
      <td>Victoria</td>
      <td>Western Europe</td>
      <td>224.7262</td>
      <td>0.0028</td>
      <td>44.9557</td>
      <td>404.4966</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
h3tukeydf.loc[h3tukeydf['reject']==True]
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
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>British Isles</td>
      <td>NSW</td>
      <td>280.1238</td>
      <td>0.0071</td>
      <td>42.2486</td>
      <td>517.9990</td>
      <td>True</td>
    </tr>
    <tr>
      <td>8</td>
      <td>British Isles</td>
      <td>Victoria</td>
      <td>306.9615</td>
      <td>0.0010</td>
      <td>104.3218</td>
      <td>509.6012</td>
      <td>True</td>
    </tr>
    <tr>
      <td>9</td>
      <td>British Isles</td>
      <td>Western Europe</td>
      <td>531.6876</td>
      <td>0.0010</td>
      <td>370.2074</td>
      <td>693.1678</td>
      <td>True</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Eastern Asia</td>
      <td>Victoria</td>
      <td>292.3065</td>
      <td>0.0033</td>
      <td>56.0970</td>
      <td>528.5160</td>
      <td>True</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Eastern Asia</td>
      <td>Western Europe</td>
      <td>517.0326</td>
      <td>0.0010</td>
      <td>315.0229</td>
      <td>719.0424</td>
      <td>True</td>
    </tr>
    <tr>
      <td>20</td>
      <td>NSW</td>
      <td>Northern Europe</td>
      <td>-300.5932</td>
      <td>0.0063</td>
      <td>-553.7320</td>
      <td>-47.4544</td>
      <td>True</td>
    </tr>
    <tr>
      <td>22</td>
      <td>NSW</td>
      <td>South America</td>
      <td>-537.9761</td>
      <td>0.0010</td>
      <td>-878.4507</td>
      <td>-197.5016</td>
      <td>True</td>
    </tr>
    <tr>
      <td>26</td>
      <td>NSW</td>
      <td>Western Europe</td>
      <td>251.5638</td>
      <td>0.0098</td>
      <td>32.8421</td>
      <td>470.2854</td>
      <td>True</td>
    </tr>
    <tr>
      <td>29</td>
      <td>North America</td>
      <td>South America</td>
      <td>-386.7846</td>
      <td>0.0011</td>
      <td>-680.0234</td>
      <td>-93.5457</td>
      <td>True</td>
    </tr>
    <tr>
      <td>33</td>
      <td>North America</td>
      <td>Western Europe</td>
      <td>402.7553</td>
      <td>0.0010</td>
      <td>268.9448</td>
      <td>536.5659</td>
      <td>True</td>
    </tr>
    <tr>
      <td>38</td>
      <td>Northern Europe</td>
      <td>Victoria</td>
      <td>327.4308</td>
      <td>0.0010</td>
      <td>107.0731</td>
      <td>547.7885</td>
      <td>True</td>
    </tr>
    <tr>
      <td>39</td>
      <td>Northern Europe</td>
      <td>Western Europe</td>
      <td>552.1569</td>
      <td>0.0010</td>
      <td>368.9344</td>
      <td>735.3795</td>
      <td>True</td>
    </tr>
    <tr>
      <td>40</td>
      <td>Scandinavia</td>
      <td>South America</td>
      <td>-339.2738</td>
      <td>0.0216</td>
      <td>-653.1004</td>
      <td>-25.4472</td>
      <td>True</td>
    </tr>
    <tr>
      <td>43</td>
      <td>Scandinavia</td>
      <td>Victoria</td>
      <td>225.5400</td>
      <td>0.0275</td>
      <td>12.4898</td>
      <td>438.5901</td>
      <td>True</td>
    </tr>
    <tr>
      <td>44</td>
      <td>Scandinavia</td>
      <td>Western Europe</td>
      <td>450.2661</td>
      <td>0.0010</td>
      <td>275.9005</td>
      <td>624.6317</td>
      <td>True</td>
    </tr>
    <tr>
      <td>45</td>
      <td>South America</td>
      <td>South-East Asia</td>
      <td>440.5338</td>
      <td>0.0027</td>
      <td>89.5674</td>
      <td>791.5001</td>
      <td>True</td>
    </tr>
    <tr>
      <td>46</td>
      <td>South America</td>
      <td>Southern Europe</td>
      <td>437.1198</td>
      <td>0.0010</td>
      <td>131.1572</td>
      <td>743.0824</td>
      <td>True</td>
    </tr>
    <tr>
      <td>47</td>
      <td>South America</td>
      <td>Victoria</td>
      <td>564.8138</td>
      <td>0.0010</td>
      <td>247.9523</td>
      <td>881.6752</td>
      <td>True</td>
    </tr>
    <tr>
      <td>48</td>
      <td>South America</td>
      <td>Western Europe</td>
      <td>789.5399</td>
      <td>0.0010</td>
      <td>497.2829</td>
      <td>1081.7970</td>
      <td>True</td>
    </tr>
    <tr>
      <td>51</td>
      <td>South-East Asia</td>
      <td>Western Europe</td>
      <td>349.0061</td>
      <td>0.0010</td>
      <td>114.2859</td>
      <td>583.7264</td>
      <td>True</td>
    </tr>
    <tr>
      <td>53</td>
      <td>Southern Europe</td>
      <td>Western Europe</td>
      <td>352.4201</td>
      <td>0.0010</td>
      <td>192.6411</td>
      <td>512.1991</td>
      <td>True</td>
    </tr>
    <tr>
      <td>54</td>
      <td>Victoria</td>
      <td>Western Europe</td>
      <td>224.7262</td>
      <td>0.0028</td>
      <td>44.9557</td>
      <td>404.4966</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
h3tukeyprepdf.head()
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
      <th>data</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>168.0</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>2</td>
      <td>174.0</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>9</td>
      <td>234.0</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>14</td>
      <td>200.0</td>
      <td>Southern Europe</td>
    </tr>
    <tr>
      <td>31</td>
      <td>153.6</td>
      <td>Southern Europe</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.barplot(data=h3tukeyprepdf,x='group',y='data',ci=68)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


```




    [Text(0, 0, 'Southern Europe'),
     Text(0, 0, 'South-East Asia'),
     Text(0, 0, 'Eastern Asia'),
     Text(0, 0, 'NSW'),
     Text(0, 0, 'North America'),
     Text(0, 0, 'Northern Europe'),
     Text(0, 0, 'British Isles'),
     Text(0, 0, 'Scandinavia'),
     Text(0, 0, 'Western Europe'),
     Text(0, 0, 'South America'),
     Text(0, 0, 'Victoria')]




![png](output_174_1.png)


### Calculate effect size (e.g. Cohen's  𝑑 )

### Report statistical power (optional)

statsmodels.stats.power:
TTestIndPower , TTestPower

# Hypothesis 4

## Question

> Does employee have a statistically significant effect on the average total spent ($) per order? If so, which employees(s)?

## Null and Alternative Hypothesis

- $H_0$: Employees do not have a significant effect on average total spent per order. 

- $H_1$: Employees do have a significant effect on average total spent per order.

## STEP 1: Determine the category/type of test based on your data

Type of data: Numerical <br>
    How many groups compared: There are 9 different employees/groups to compare.
    - So we'll use ANOVA & Tukey tests since there are more than two groups. 

### Put data in df


```python
cur.execute("""SELECT *
                FROM OrderDetail
                """)
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id', 'OrderId', 'ProductId', 'UnitPrice', 'Quantity', 'Discount']




```python
cur.execute("""SELECT *
                FROM `Order`
                """)
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id',
     'CustomerId',
     'EmployeeId',
     'OrderDate',
     'RequiredDate',
     'ShippedDate',
     'ShipVia',
     'Freight',
     'ShipName',
     'ShipAddress',
     'ShipCity',
     'ShipRegion',
     'ShipPostalCode',
     'ShipCountry']




```python
cur.execute("""SELECT *
                FROM Employee
                """)
col_names=[x[0] for x in cur.description]
col_names
```




    ['Id',
     'LastName',
     'FirstName',
     'Title',
     'TitleOfCourtesy',
     'BirthDate',
     'HireDate',
     'Address',
     'City',
     'Region',
     'PostalCode',
     'Country',
     'HomePhone',
     'Extension',
     'Photo',
     'Notes',
     'ReportsTo',
     'PhotoPath']




```python
cur.execute("""SELECT od.OrderId, od.UnitPrice, od.Quantity, e.Id, e.LastName, e.FirstName
                FROM OrderDetail od
                JOIN `Order` o ON od.OrderId = o.Id                
                JOIN Employee e ON o.EmployeeId = e.Id                
""")
col_names=[x[0] for x in cur.description]
col_names
```




    ['OrderId', 'UnitPrice', 'Quantity', 'Id', 'LastName', 'FirstName']




```python
h4df = pd.DataFrame(cur.fetchall(), columns=col_names)
h4df
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
      <th>OrderId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Id</th>
      <th>LastName</th>
      <th>FirstName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248</td>
      <td>14.00</td>
      <td>12</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248</td>
      <td>9.80</td>
      <td>10</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248</td>
      <td>34.80</td>
      <td>5</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249</td>
      <td>18.60</td>
      <td>9</td>
      <td>6</td>
      <td>Suyama</td>
      <td>Michael</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249</td>
      <td>42.40</td>
      <td>40</td>
      <td>6</td>
      <td>Suyama</td>
      <td>Michael</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077</td>
      <td>33.25</td>
      <td>2</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077</td>
      <td>17.00</td>
      <td>1</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077</td>
      <td>15.00</td>
      <td>2</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077</td>
      <td>7.75</td>
      <td>4</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077</td>
      <td>13.00</td>
      <td>2</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 6 columns</p>
</div>




```python
# we have same number of unique first and last names so there are no people with same last name that could 
# be accidentally grouped as one person. 
h4df['FirstName'].nunique()
```




    9




```python
h4df['LastName'].nunique()
```




    9




```python
## create total spent
h4df["Total Spent"] = h4df['UnitPrice'] * h4df['Quantity']
h4df
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
      <th>OrderId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Id</th>
      <th>LastName</th>
      <th>FirstName</th>
      <th>Total Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248</td>
      <td>14.00</td>
      <td>12</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
      <td>168.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248</td>
      <td>9.80</td>
      <td>10</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
      <td>98.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248</td>
      <td>34.80</td>
      <td>5</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
      <td>174.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249</td>
      <td>18.60</td>
      <td>9</td>
      <td>6</td>
      <td>Suyama</td>
      <td>Michael</td>
      <td>167.4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249</td>
      <td>42.40</td>
      <td>40</td>
      <td>6</td>
      <td>Suyama</td>
      <td>Michael</td>
      <td>1696.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2150</td>
      <td>11077</td>
      <td>33.25</td>
      <td>2</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
      <td>66.5</td>
    </tr>
    <tr>
      <td>2151</td>
      <td>11077</td>
      <td>17.00</td>
      <td>1</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>2152</td>
      <td>11077</td>
      <td>15.00</td>
      <td>2</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
      <td>30.0</td>
    </tr>
    <tr>
      <td>2153</td>
      <td>11077</td>
      <td>7.75</td>
      <td>4</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>2154</td>
      <td>11077</td>
      <td>13.00</td>
      <td>2</td>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 7 columns</p>
</div>




```python
ax = sns.barplot(data=h4df, x='LastName', y='Total Spent', ci=68)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


```




    [Text(0, 0, 'Buchanan'),
     Text(0, 0, 'Suyama'),
     Text(0, 0, 'Peacock'),
     Text(0, 0, 'Leverling'),
     Text(0, 0, 'Dodsworth'),
     Text(0, 0, 'Davolio'),
     Text(0, 0, 'Callahan'),
     Text(0, 0, 'Fuller'),
     Text(0, 0, 'King')]




![png](output_194_1.png)


### put data into dict


```python
h4dict = {}
for name in h4df['LastName'].unique():
    h4dict[name] = h4df.groupby('LastName').get_group(name)['Total Spent']
    
h4dict
```




    {'Buchanan': 0       168.0
     1        98.0
     2       174.0
     17       54.0
     18      403.2
             ...  
     1829    488.6
     1830    312.5
     1831    285.0
     1832    816.0
     2053    210.0
     Name: Total Spent, Length: 117, dtype: float64, 'Suyama': 3        167.4
     4       1696.0
     47       532.0
     48       192.5
     61        48.0
              ...  
     2023      94.5
     2024     665.0
     2025     344.0
     2055      37.5
     2056    1272.0
     Name: Total Spent, Length: 168, dtype: float64, 'Peacock': 5         77.0
     6       1484.0
     7        252.0
     11      2592.0
     12        50.0
              ...  
     2119     357.5
     2120    4322.5
     2127     500.0
     2128     465.0
     2129      92.0
     Name: Total Spent, Length: 420, dtype: float64, 'Leverling': 8       100.8
     9       234.0
     10      336.0
     14      200.0
     15      604.8
             ...  
     2068    285.0
     2081     45.0
     2093    420.0
     2094    736.0
     2095    289.5
     Name: Total Spent, Length: 321, dtype: float64, 'Dodsworth': 20       304.0
     21       486.5
     22       380.0
     23      1320.0
     43       834.0
              ...  
     1999     322.0
     2000    1080.0
     2082      30.0
     2083     714.0
     2084     114.0
     Name: Total Spent, Length: 107, dtype: float64, 'Davolio': 29       760.0
     30      1105.0
     31       153.6
     59       456.0
     60       920.0
              ...  
     2150      66.5
     2151      17.0
     2152      30.0
     2153      31.0
     2154      26.0
     Name: Total Spent, Length: 345, dtype: float64, 'Callahan': 40       204.0
     41       360.0
     42        60.8
     55       990.0
     56       111.2
              ...  
     2108    1656.0
     2109     364.0
     2124     190.0
     2125     360.0
     2126      36.0
     Name: Total Spent, Length: 260, dtype: float64, 'Fuller': 49      936.0
     50      240.0
     76      728.0
     77      472.8
     83       43.2
             ...  
     2112    380.0
     2113    523.5
     2114    250.0
     2121    210.0
     2122     90.0
     Name: Total Spent, Length: 241, dtype: float64, 'King': 109     240.00
     110     239.40
     147     588.00
     148     504.00
     149     150.00
              ...  
     2077    390.00
     2103     52.35
     2104    386.40
     2105    490.00
     2123    244.30
     Name: Total Spent, Length: 176, dtype: float64}



## STEP 2: Do we meet the assumptions of the chosen test?

ANOVA tukey

- No significant outliers
- Normality
- Equal Variance

### 0. Check for & Remove Outliers


```python
fig,ax=plt.subplots(figsize=(8,5))
for name,values in h4dict.items():
    sns.distplot(values,label=name, ax=ax)
    
    
ax.legend()
ax.set(title='Total Spent from Sale by Employee', ylabel='Density')
```




    [Text(0, 0.5, 'Density'), Text(0.5, 1.0, 'Total Spent from Sale by Employee')]




![png](output_200_1.png)


Looks like there are some outliers.  


```python
for name,values in h4dict.items():
    idx_outs = find_outliers_Z(values)
    print(f"Found {idx_outs.sum()} outliers using Z-score method for {name}.")
    h4dict[name] = values[~idx_outs]
    
```

    Found 3 outliers using Z-score method for Buchanan.
    Found 4 outliers using Z-score method for Suyama.
    Found 6 outliers using Z-score method for Peacock.
    Found 4 outliers using Z-score method for Leverling.
    Found 2 outliers using Z-score method for Dodsworth.
    Found 4 outliers using Z-score method for Davolio.
    Found 4 outliers using Z-score method for Callahan.
    Found 5 outliers using Z-score method for Fuller.
    Found 5 outliers using Z-score method for King.


### 1. Test Assumption of Normality


```python
for key,values in h4dict.items():
    stat,h4p = stats.normaltest(values)
    print(f"Group {key} Normaltest p-value={round(h4p,4)}")
    sig = 'is NOT' if h4p<.05 else 'IS'

    print(f"\t-The data {sig} normal.")
```

    Group Buchanan Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Suyama Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Peacock Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Leverling Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Dodsworth Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Davolio Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Callahan Normaltest p-value=0.0
    	-The data is NOT normal.
    Group Fuller Normaltest p-value=0.0
    	-The data is NOT normal.
    Group King Normaltest p-value=0.0
    	-The data is NOT normal.


### 1B. We don't have normal data:

So in order to ignore non-normality, if we have 2-9 groups, each group needs n >= 15.
If we have 10-12 groups, we need each group to have n>20. We have 9 groups. 


```python
for name in h4dict.keys():
    region = name
    print(f"{region}: ", len(h4dict[name]))
```

    Buchanan:  114
    Suyama:  164
    Peacock:  414
    Leverling:  317
    Dodsworth:  105
    Davolio:  341
    Callahan:  256
    Fuller:  236
    King:  171


All 9 group sizes (n) are bigger than required 20 for each group so we can safely ignore normality assumption. <br>
So we move on to equal variance assumption.

### 2. Test for Equal Variance


```python
# put data into lists for levene's test args
h4data = []
for key,vals in h4dict.items():
    h4data.append(vals)
```


```python
h4data[1]
```




    3        167.4
    4       1696.0
    47       532.0
    48       192.5
    61        48.0
             ...  
    2023      94.5
    2024     665.0
    2025     344.0
    2055      37.5
    2056    1272.0
    Name: Total Spent, Length: 164, dtype: float64




```python
stat,h4p = stats.levene(*h4data)
print(f"Levene's Test for Equal Variance p-value={round(h4p,4)}")
sig = 'do NOT' if h4p<.05 else 'DO'

print(f"\t-The groups {sig} have equal variance.")
```

    Levene's Test for Equal Variance p-value=0.0085
    	-The groups do NOT have equal variance.


### Failed the assumption of equal variance:

Since we don't have equal variance and data is not normal, we need to use a non-parametric version of a t-test. 
 - The non-parametric version for ANOVA is Kruskal-Wallis.
 - Works from medians instead of means. 
 - scipy.stats.kruskal

## STEP 3: Interpret Result & Post-Hoc Tests

### Perform hypothesis test from summary table above to get your p-value.

- If p value is < $\alpha$:

    - Reject the null hypothesis.
    - Calculate effect size (e.g. Cohen's $d$)
- If p<.05 AND you have multiple groups (i.e. ANOVA)

    - Must run a pairwise Tukey's test to know which groups were significantly different.
    - Tukey pairwise comparison test
    - statsmodels.stats.multicomp.pairwise_tukeyhsd
- Report statistical power (optional)


```python
stat,h4p = stats.kruskal(*h4data)
print(f"Kruskal-Wallis p-value={round(h4p,4)}")
```

    Kruskal-Wallis p-value=0.2356


Kruskal-Wallis p val shows that the null hypothesis should NOT be rejected and employee does have significant effect on average total spent per order. 

### Could calc effect size to then calc power to see if there's a possibility of type 2 errors.


```python

```
