```python
# import libraries


import numpy as np
import pandas as pd #data processing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Visualization and plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go



# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

```


```python
# loading the data

df_census = pd.read_csv('D:/Mahdieh_CourseUniversity/University_courses/ALY6020\Module_1/Project/Project/adult-all.csv')

```

Exploratory Data Analysis


```python
#check head and tail of data set

df_census.head()
df_census.tail()
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
      <th>Age</th>
      <th>WorkClass</th>
      <th>Fnlwgt</th>
      <th>Education</th>
      <th>EducationNum</th>
      <th>MaritalStatus</th>
      <th>Occupation</th>
      <th>Relashonship</th>
      <th>Race</th>
      <th>Sex</th>
      <th>CapitalGain</th>
      <th>CapitalLoss</th>
      <th>HoursPerWeek</th>
      <th>NativeCountry</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48837</th>
      <td>39</td>
      <td>Private</td>
      <td>215419</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Divorced</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>48838</th>
      <td>64</td>
      <td>?</td>
      <td>321403</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Other-relative</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>48839</th>
      <td>38</td>
      <td>Private</td>
      <td>374983</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>48840</th>
      <td>44</td>
      <td>Private</td>
      <td>83891</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Divorced</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>5455</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>48841</th>
      <td>35</td>
      <td>Self-emp-inc</td>
      <td>182148</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
#number of columns and the length of the dataset

df_census.shape
```




    (48842, 15)




```python
#descriptive analysis 

df_census.describe()
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
      <th>Age</th>
      <th>Fnlwgt</th>
      <th>EducationNum</th>
      <th>CapitalGain</th>
      <th>CapitalLoss</th>
      <th>HoursPerWeek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>48842.000000</td>
      <td>4.884200e+04</td>
      <td>48842.000000</td>
      <td>48842.000000</td>
      <td>48842.000000</td>
      <td>48842.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.643585</td>
      <td>1.896641e+05</td>
      <td>10.078089</td>
      <td>1079.067626</td>
      <td>87.502314</td>
      <td>40.422382</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.710510</td>
      <td>1.056040e+05</td>
      <td>2.570973</td>
      <td>7452.019058</td>
      <td>403.004552</td>
      <td>12.391444</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>1.228500e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>1.175505e+05</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>1.781445e+05</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>2.376420e+05</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>1.490400e+06</td>
      <td>16.000000</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>



It Seems in the column capitalGain and capitalLoss more than 90% values are zeros.hence , the variables highly skewed


```python
#Check type of variables and find null values

df_census.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 48842 entries, 0 to 48841
    Data columns (total 15 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   Age            48842 non-null  int64 
     1   WorkClass      48842 non-null  object
     2   Fnlwgt         48842 non-null  int64 
     3   Education      48842 non-null  object
     4   EducationNum   48842 non-null  int64 
     5   MaritalStatus  48842 non-null  object
     6   Occupation     48842 non-null  object
     7   Relashonship   48842 non-null  object
     8   Race           48842 non-null  object
     9   Sex            48842 non-null  object
     10  CapitalGain    48842 non-null  int64 
     11  CapitalLoss    48842 non-null  int64 
     12  HoursPerWeek   48842 non-null  int64 
     13  NativeCountry  48842 non-null  object
     14  Salary         48842 non-null  object
    dtypes: int64(6), object(9)
    memory usage: 5.6+ MB
    


```python
#Check null values
df_census.isnull().values.any()
```




    False




```python
df_census.isnull().sum()
```




    Age              0
    WorkClass        0
    Fnlwgt           0
    Education        0
    EducationNum     0
    MaritalStatus    0
    Occupation       0
    Relashonship     0
    Race             0
    Sex              0
    CapitalGain      0
    CapitalLoss      0
    HoursPerWeek     0
    NativeCountry    0
    Salary           0
    dtype: int64




```python
# Find the Number of Categories for Categorical variables / WorkClass

df_census['WorkClass'].unique()
```




    array(['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
           'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
          dtype=object)




```python
# Find the Number of Categories for Categorical variables / Education

df_census['Education'].unique()
```




    array(['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
           'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
           '5th-6th', '10th', '1st-4th', 'Preschool', '12th'], dtype=object)




```python
# Find the Number of Categories for Categorical variables / MaritalStatus

df_census['MaritalStatus'].unique()
```




    array(['Never-married', 'Married-civ-spouse', 'Divorced',
           'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
           'Widowed'], dtype=object)




```python
# Find the Number of Categories for Categorical variables / Occupation

df_census['Occupation'].unique()
```




    array(['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
           'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
           'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
           'Tech-support', '?', 'Protective-serv', 'Armed-Forces',
           'Priv-house-serv'], dtype=object)




```python
# Find the Number of Categories for Categorical variables / Relashonship

df_census['Relashonship'].unique()
```




    array(['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
           'Other-relative'], dtype=object)




```python
# Find the Number of Categories for Categorical variables / Race

df_census['Race'].unique()
```




    array(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
           'Other'], dtype=object)




```python
# Find the Number of Categories for Categorical variables / Sex

df_census['Sex'].unique()
```




    array(['Male', 'Female'], dtype=object)




```python
# Replace ? with Not-Known in WorkClass and Occupation variables

df_census.WorkClass = df_census.WorkClass.replace({'?' : 'Not-Known'})
df_census.Occupation = df_census.Occupation.replace({'?' : 'Not-Known'})
df_census.NativeCountry = df_census.NativeCountry.replace({'?' : 'Not-Known'})
```


```python
# Separating categorical Variables

cat_variables = [x for x in df_census.columns if df_census[x].dtype == "object"]
cat_variables
```




    ['WorkClass',
     'Education',
     'MaritalStatus',
     'Occupation',
     'Relashonship',
     'Race',
     'Sex',
     'NativeCountry',
     'Salary']




```python
# Separating Numeric Variables

num_variables = [x for x in df_census.columns if x not in cat_variables]
num_variables
```




    ['Age', 'Fnlwgt', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek']




```python
#Checking the unique values and their counts in each categorical variable

for i in cat_variables:
    print("------"+i+"---------")
    print(df_census[i].value_counts(normalize=True))
    print("\n")
```

    ------WorkClass---------
    Private             0.694198
    Self-emp-not-inc    0.079071
    Local-gov           0.064207
    Not-Known           0.057307
    State-gov           0.040559
    Self-emp-inc        0.034704
    Federal-gov         0.029319
    Without-pay         0.000430
    Never-worked        0.000205
    Name: WorkClass, dtype: float64
    
    
    ------Education---------
    HS-grad         0.323164
    Some-college    0.222718
    Bachelors       0.164305
    Masters         0.054400
    Assoc-voc       0.042197
    11th            0.037099
    Assoc-acdm      0.032779
    10th            0.028439
    7th-8th         0.019553
    Prof-school     0.017075
    9th             0.015478
    12th            0.013452
    Doctorate       0.012162
    5th-6th         0.010421
    1st-4th         0.005057
    Preschool       0.001699
    Name: Education, dtype: float64
    
    
    ------MaritalStatus---------
    Married-civ-spouse       0.458192
    Never-married            0.329982
    Divorced                 0.135805
    Separated                0.031325
    Widowed                  0.031080
    Married-spouse-absent    0.012858
    Married-AF-spouse        0.000758
    Name: MaritalStatus, dtype: float64
    
    
    ------Occupation---------
    Prof-specialty       0.126367
    Craft-repair         0.125138
    Exec-managerial      0.124606
    Adm-clerical         0.114881
    Sales                0.112690
    Other-service        0.100794
    Machine-op-inspct    0.061873
    Not-Known            0.057512
    Transport-moving     0.048217
    Handlers-cleaners    0.042423
    Farming-fishing      0.030507
    Tech-support         0.029606
    Protective-serv      0.020126
    Priv-house-serv      0.004955
    Armed-Forces         0.000307
    Name: Occupation, dtype: float64
    
    
    ------Relashonship---------
    Husband           0.403669
    Not-in-family     0.257627
    Own-child         0.155215
    Unmarried         0.104930
    Wife              0.047725
    Other-relative    0.030834
    Name: Relashonship, dtype: float64
    
    
    ------Race---------
    White                 0.855043
    Black                 0.095922
    Asian-Pac-Islander    0.031100
    Amer-Indian-Eskimo    0.009623
    Other                 0.008313
    Name: Race, dtype: float64
    
    
    ------Sex---------
    Male      0.668482
    Female    0.331518
    Name: Sex, dtype: float64
    
    
    ------NativeCountry---------
    United-States                 0.897424
    Mexico                        0.019471
    Not-Known                     0.017546
    Philippines                   0.006040
    Germany                       0.004218
    Puerto-Rico                   0.003767
    Canada                        0.003726
    El-Salvador                   0.003173
    India                         0.003092
    Cuba                          0.002825
    England                       0.002600
    China                         0.002498
    South                         0.002355
    Jamaica                       0.002170
    Italy                         0.002150
    Dominican-Republic            0.002109
    Japan                         0.001884
    Guatemala                     0.001802
    Poland                        0.001781
    Vietnam                       0.001761
    Columbia                      0.001740
    Haiti                         0.001536
    Portugal                      0.001372
    Taiwan                        0.001331
    Iran                          0.001208
    Greece                        0.001003
    Nicaragua                     0.001003
    Peru                          0.000942
    Ecuador                       0.000921
    France                        0.000778
    Ireland                       0.000758
    Hong                          0.000614
    Thailand                      0.000614
    Cambodia                      0.000573
    Trinadad&Tobago               0.000553
    Laos                          0.000471
    Yugoslavia                    0.000471
    Outlying-US(Guam-USVI-etc)    0.000471
    Scotland                      0.000430
    Honduras                      0.000409
    Hungary                       0.000389
    Holand-Netherlands            0.000020
    Name: NativeCountry, dtype: float64
    
    
    ------Salary---------
    <=50K    0.760718
    >50K     0.239282
    Name: Salary, dtype: float64
    
    
    


```python

```


```python
#Checking Target variable / Salary

df_census.Salary.value_counts(normalize=True)
```




    <=50K    0.760718
    >50K     0.239282
    Name: Salary, dtype: float64




```python
# visualize Salary with its count

plt.figure(figsize=(6,4))
df_census.Salary.value_counts().plot(kind = "bar")
plt.xticks(rotation = 0)
plt.show()
```


    
![png](output_23_0.png)
    



```python
# visualize categorical variables with their count

for i in cat_variables:
    plt.figure(figsize=(10,13))
    plt.title(i)
    sns.countplot(x = df_census[i] , order = df_census[i].value_counts().index )
    
    plt.xticks(rotation = 90, fontsize = 7)
    plt.show()

    
```


    
![png](output_24_0.png)
    



    
![png](output_24_1.png)
    



    
![png](output_24_2.png)
    



    
![png](output_24_3.png)
    



    
![png](output_24_4.png)
    



    
![png](output_24_5.png)
    



    
![png](output_24_6.png)
    



    
![png](output_24_7.png)
    



    
![png](output_24_8.png)
    



```python

# Correlation matrix between numeric variables

plt.figure(figsize=(8,4)) 
sns.heatmap(df_census.corr(),annot=True) #draws  heatmap with input as the correlation matrix calculted by(df_census.corr())
plt.show()
```


    
![png](output_25_0.png)
    



```python
#find correlation between categorical variables and salary using chi test

# Separating categorical Variables
df_cat = pd.DataFrame(data = df_census.dtypes, columns = ['a']).reset_index()
cat_var = list(df_cat['index'].loc[df_cat['a'] == 'object'])

df_cat = df_census[cat_var]

## Let us jump to Chi-Square test
## Creating all possible combinations between the above two variables list

from itertools import product

cat_var_prod = list(product(cat1_variables,cat2_variables, repeat = 1))

#######################################################

import scipy.stats as ss

result = []
for i in cat_var_prod:
    if i[0] != i[1]:
        result.append((i[0],i[1],list(ss.chi2_contingency(pd.crosstab(df_cat[i[0]], df_cat[i[1]])))[1]))
        
result
#######################################################

chi_test_output = pd.DataFrame(result, columns = ['var1', 'var1', 'coeff'])
chi_test_output.tail(8)
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
      <th>var1</th>
      <th>var1</th>
      <th>coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>Salary</td>
      <td>WorkClass</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Salary</td>
      <td>Education</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Salary</td>
      <td>MaritalStatus</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Salary</td>
      <td>Occupation</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Salary</td>
      <td>Relashonship</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Salary</td>
      <td>Race</td>
      <td>4.284378e-104</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Salary</td>
      <td>Sex</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Salary</td>
      <td>NativeCountry</td>
      <td>1.035618e-70</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check correlation for all data set variables using converting category to dummy variables

#convert categorical to numerical 

df_census['WorkClass'] = LabelEncoder().fit_transform(df_census['WorkClass'])

df_census['Education'] = LabelEncoder().fit_transform(df_census['Education'])

df_census['MaritalStatus'] = LabelEncoder().fit_transform(df_census['MaritalStatus'])

df_census['Occupation'] = LabelEncoder().fit_transform(df_census['Occupation'])

df_census['Relashonship'] = LabelEncoder().fit_transform(df_census['Relashonship'])

df_census['Race'] = LabelEncoder().fit_transform(df_census['Race'])

df_census['Sex'] = LabelEncoder().fit_transform(df_census['Sex'])

df_census['NativeCountry'] = LabelEncoder().fit_transform(df_census['NativeCountry'])

df_census['Salary'] = LabelEncoder().fit_transform(df_census['Salary'])


# Correlation matrix

plt.figure(figsize=(12,10)) 
sns.heatmap(df_census.corr(),annot=True) #draws  heatmap with input as the correlation matrix calculted by(df_census.corr())
plt.show()
```


    
![png](output_27_0.png)
    



```python
#distribution of numerical variables

df_census.hist(edgecolor='black', linewidth=0.75)
fig=plt.gcf()
fig.set_size_inches(8,10)
plt.show()
```


    
![png](output_28_0.png)
    


since capitalGain and capitalLoss have more than 90% zero I think it is not give us meaningfull information, hence I ignore them. Also, I ignore Fniwgt:final weight as well since it is the (estimated) number of people each row in the data represents.


```python

df_census_selected = df_census.drop(columns=['Fnlwgt', 'CapitalGain' , 'CapitalLoss'], axis=1)
```


```python
#check for outliers with selected numerical variables 

plt.figure(figsize=(8,6))
df_census_selected.boxplot(grid = False)
```




    <AxesSubplot: >




    
![png](output_31_1.png)
    


Visualization


```python
# Visualization of income based on WorkClass /Countplot
    
plt.figure(figsize=(11,6))

g = sns.countplot(x = 'WorkClass', hue = 'Salary',data = df_census_selected, order = df_census_selected['WorkClass'].value_counts().index,palette=['#24b1d1', '#ae24d1'])
sns.move_legend(g, "upper right", bbox_to_anchor=(1.11, 1.11), title='Salary')
plt.title("Income By WorkClass")
```




    Text(0.5, 1.0, 'Income By WorkClass')




    
![png](output_33_1.png)
    



```python

## Visualization of income based on WorkClass /histplot

plt.figure(figsize=(12,6))
s = sns.histplot(df_census_selected, x="WorkClass", hue="Salary", multiple="stack", shrink=.6,palette=['#20B2AA', '#FFD700'])
sns.move_legend(s, "upper right", bbox_to_anchor=(1.13, 1.13), title='Salary')
plt.title("Income By WorkClass")

```




    Text(0.5, 1.0, 'Income By WorkClass')




    
![png](output_34_1.png)
    



```python
# Visualization of income based on Education /Countplot
    
plt.figure(figsize=(10,9))

g = sns.countplot(x = 'Education', hue = 'Salary',data = df_census_selected, order = df_census_selected['Education'].value_counts().index,palette=['#24b1d1', '#ae24d1'])
sns.move_legend(g, "upper right", bbox_to_anchor=(1.11, 1.11), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By Education")
```




    Text(0.5, 1.0, 'Income By Education')




    
![png](output_35_1.png)
    



```python
## Visualization of income based on Education /histplot

plt.figure(figsize=(15,6))
s = sns.histplot(df_census_selected, x="Education", hue="Salary", multiple="stack", palette=['#20B2AA', '#FFD700'], shrink=.6)
sns.move_legend(s, "upper right", bbox_to_anchor=(1.13, 1.13), title='Salary')
plt.xticks(fontsize = 7)
plt.title("Income By Education")

```




    Text(0.5, 1.0, 'Income By Education')




    
![png](output_36_1.png)
    



```python
## Visualization of income based on Sex /histplot

plt.figure(figsize=(8,6))
s = sns.histplot(df_census_selected, x="Sex", hue="Salary", multiple="stack", palette=['#20B2AA', '#FFD700'], shrink=.3)
sns.move_legend(s, "upper right", bbox_to_anchor=(1.13, 1.13), title='Salary')
plt.xticks(fontsize = 7)
plt.title("Income By Sex")
```




    Text(0.5, 1.0, 'Income By Sex')




    
![png](output_37_1.png)
    



```python
# Visualization of income based on Sex /Countplot
    
plt.figure(figsize=(8,6))

g = sns.countplot(x = 'Sex', hue = 'Salary',data = df_census_selected, order = df_census_selected['Sex'].value_counts().index,palette=['#24b1d1', '#ae24d1'])
sns.move_legend(g, "upper right", bbox_to_anchor=(1.11, 1.11), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By Sex")
```




    Text(0.5, 1.0, 'Income By Sex')




    
![png](output_38_1.png)
    



```python
## Visualization of income based on Occupation /histplot

plt.figure(figsize=(12,11))
s = sns.histplot(df_census_selected, x="Occupation", hue="Salary", multiple="stack", palette=['#20B2AA', '#FFD700'], shrink=.5)
sns.move_legend(s, "upper right", bbox_to_anchor=(1.13, 1.13), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By Occupation") 
```




    Text(0.5, 1.0, 'Income By Occupation')




    
![png](output_39_1.png)
    



```python
# Visualization of income based on Occupation /Countplot
    
plt.figure(figsize=(10,9))

g = sns.countplot(x = 'Occupation', hue = 'Salary',data = df_census_selected, order = df_census_selected['Occupation'].value_counts().index, palette=['#24b1d1', '#ae24d1'])
sns.move_legend(g, "upper right", bbox_to_anchor=(1.11, 1.11), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By Occupation")
```




    Text(0.5, 1.0, 'Income By Occupation')




    
![png](output_40_1.png)
    



```python
## Visualization of income based on race /histplot

plt.figure(figsize=(11,12))
s = sns.histplot(df_census_selected, x="Race", hue="Salary", multiple="stack", palette=['#20B2AA', '#FFD700'], shrink=.5)
sns.move_legend(s, "upper right", bbox_to_anchor=(1.13, 1.13), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By Race") 
```




    Text(0.5, 1.0, 'Income By Race')




    
![png](output_41_1.png)
    



```python
# Visualization of income based on Race /Countplot
    
plt.figure(figsize=(10,9))

g = sns.countplot(x = 'Race', hue = 'Salary',data = df_census_selected, order = df_census_selected['Race'].value_counts().index, palette=['#24b1d1', '#ae24d1'])
sns.move_legend(g, "upper right", bbox_to_anchor=(1.11, 1.11), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By Race")
```




    Text(0.5, 1.0, 'Income By Race')




    
![png](output_42_1.png)
    



```python
## Visualization of income based on MaritalStatus /histplot

plt.figure(figsize=(9,13))
s = sns.histplot(df_census_selected, x="MaritalStatus", hue="Salary", multiple="stack", palette=['#20B2AA', '#FFD700'], shrink=.5)
sns.move_legend(s, "upper right", bbox_to_anchor=(1.13, 1.13), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By MaritalStatus") 
```




    Text(0.5, 1.0, 'Income By MaritalStatus')




    
![png](output_43_1.png)
    



```python
# Visualization of income based on MaritalStatus /Countplot
    
plt.figure(figsize=(10,9))

g = sns.countplot(x = 'MaritalStatus', hue = 'Salary',data = df_census_selected, order = df_census_selected['MaritalStatus'].value_counts().index, palette=['#24b1d1', '#ae24d1'])
sns.move_legend(g, "upper right", bbox_to_anchor=(1.11, 1.11), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By MaritalStatus")
```




    Text(0.5, 1.0, 'Income By MaritalStatus')




    
![png](output_44_1.png)
    



```python
## Visualization of income based on NativeCountry /histplot

plt.figure(figsize=(12,11))
s = sns.histplot(df_census_selected, x="NativeCountry", hue="Salary", multiple="stack", palette=['#20B2AA', '#FFD700'], shrink=.7)
sns.move_legend(s, "upper right", bbox_to_anchor=(1.13, 1.13), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By NativeCountry") 
```




    Text(0.5, 1.0, 'Income By NativeCountry')




    
![png](output_45_1.png)
    



```python
# Visualization of income based on NativeCountry /Countplot
    
plt.figure(figsize=(12,9))

g = sns.countplot(x = 'NativeCountry', hue = 'Salary',data = df_census_selected, order = df_census_selected['NativeCountry'].value_counts().index, palette=['#24b1d1', '#ae24d1'])
sns.move_legend(g, "upper right", bbox_to_anchor=(1.11, 1.11), title='Salary')
plt.xticks(rotation = 90, fontsize = 7)
plt.title("Income By NativeCountry")
```




    Text(0.5, 1.0, 'Income By NativeCountry')




    
![png](output_46_1.png)
    


Feature Selection:
    
since capitalGain and capitalLoss have more than 90% zero I think it is not give us meaningfull information, hence I ignore them. Also, I ignore Fniwgt:final weight as well since it is the (estimated) number of people each row in the data represents.


Perform process of KNN Machine Learning Model for predicting new Income based on other features.

I perform Model for two datasets. the original dataset. and the Feature Selected Data set


Perform Model with Feature Selected Data set:


```python
#convert categorical to numerical 

df_census_selected['WorkClass'] = LabelEncoder().fit_transform(df_census_selected['WorkClass'])

df_census_selected['Education'] = LabelEncoder().fit_transform(df_census_selected['Education'])

df_census_selected['MaritalStatus'] = LabelEncoder().fit_transform(df_census_selected['MaritalStatus'])

df_census_selected['Occupation'] = LabelEncoder().fit_transform(df_census_selected['Occupation'])

df_census_selected['Relashonship'] = LabelEncoder().fit_transform(df_census_selected['Relashonship'])

df_census_selected['Race'] = LabelEncoder().fit_transform(df_census_selected['Race'])

df_census_selected['Sex'] = LabelEncoder().fit_transform(df_census_selected['Sex'])

df_census_selected['NativeCountry'] = LabelEncoder().fit_transform(df_census_selected['NativeCountry'])

df_census_selected['Salary'] = LabelEncoder().fit_transform(df_census_selected['Salary'])
```


```python
# seperating features , y is predicted variable and x is predictors

x= df_census_selected.iloc[:,0:10]
y= df_census_selected.iloc[:,-1]
```


```python
# splitting Data set into test and train

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
```


```python
#selecting different ks in range [1,15] and choose one with highest accuracy

k_range = list(range(1,16))
accuracy = []

for i in k_range:
    
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)
    y_predict = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test,y_predict))
    
plt.figure(figsize=(9,6)) 
plt.plot(k_range, accuracy)
```




    [<matplotlib.lines.Line2D at 0x16d597f2d00>]




    
![png](output_53_1.png)
    



```python
accuracy
```




    [0.7721968197638709,
     0.7925339520917218,
     0.7925339520917218,
     0.8022930457926705,
     0.7970381491844674,
     0.80584180713847,
     0.8046816351600354,
     0.8084351327373234,
     0.8044086535180509,
     0.8080939056848427,
     0.8079574148638504,
     0.8082986419163312,
     0.8084351327373234,
     0.8089810960212925,
     0.8082986419163312]



Perform Model with Original Data set with all features:


```python
#convert categorical to numerical 

df_census['WorkClass'] = LabelEncoder().fit_transform(df_census['WorkClass'])

df_census['Education'] = LabelEncoder().fit_transform(df_census['Education'])

df_census['MaritalStatus'] = LabelEncoder().fit_transform(df_census['MaritalStatus'])

df_census['Occupation'] = LabelEncoder().fit_transform(df_census['Occupation'])

df_census['Relashonship'] = LabelEncoder().fit_transform(df_census['Relashonship'])

df_census['Race'] = LabelEncoder().fit_transform(df_census['Race'])

df_census['Sex'] = LabelEncoder().fit_transform(df_census['Sex'])

df_census['NativeCountry'] = LabelEncoder().fit_transform(df_census['NativeCountry'])

df_census['Salary'] = LabelEncoder().fit_transform(df_census['Salary'])
```


```python
# seperating features , y is predicted variable and x is predictors

x= df_census.iloc[:,0:13]
y= df_census.iloc[:,-1]
```


```python
# splitting Data set into test and train

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
```


```python
#selecting different ks in range [1,15] and choose one with highest accuracy

k_range = list(range(1,16))
accuracy = []

for i in k_range:
    
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)
    y_predict = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test,y_predict))
    
plt.figure(figsize=(9,6)) 
plt.plot(k_range, accuracy)
```




    [<matplotlib.lines.Line2D at 0x16d596c0d90>]




    
![png](output_59_1.png)
    



```python
accuracy
```




    [0.7344571077595031,
     0.7877567733569918,
     0.7644850883778066,
     0.7923292158602334,
     0.7787483791715007,
     0.7947178052275985,
     0.787893264177984,
     0.7973793762369481,
     0.7944448235856139,
     0.7996314747833209,
     0.796901658363475,
     0.7992220023203439,
     0.7966969221319866,
     0.801610591687709,
     0.801610591687709]



Model Evaluating :

 *  Original dataset Optimal K is 14 and Accuracy 0.7982665665733979
 *  Feature Selected Data set Optimal K is 10 and Accuracy 0.8163516003548761


```python

```
