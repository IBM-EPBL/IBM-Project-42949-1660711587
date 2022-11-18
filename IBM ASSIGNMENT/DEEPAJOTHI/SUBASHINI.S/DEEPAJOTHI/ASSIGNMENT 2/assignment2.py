{\rtf1\ansi\ansicpg1252\deff0\deflang1033{\fonttbl{\f0\fnil\fcharset0 Calibri;}}
{\*\generator Msftedit 5.41.21.2510;}\viewkind4\uc1\pard\sa200\sl276\slmult1\lang9\f0\fs22 1.Download the dataset\par
2.Load the dataset\par
import pandas as pd\par
import numpy as np\par
import seaborn as sns\par
import matplotlib.pyplot as plt\par
file=pd.read_csv("Churn_Modelling.csv")\par
df=pd.DataFrame(file)\par
df.head()\par
RowNumber\tab CustomerId\tab Surname\tab CreditScore\tab Geography\tab Gender\tab Age\tab Tenure\tab Balance\tab NumOfProducts\tab HasCrCard\tab IsActiveMember\tab EstimatedSalary\tab Exited\par
0\tab 1\tab 15634602\tab Hargrave\tab 619\tab France\tab Female\tab 42\tab 2\tab 0.00\tab 1\tab 1\tab 1\tab 101348.88\tab 1\par
1\tab 2\tab 15647311\tab Hill\tab 608\tab Spain\tab Female\tab 41\tab 1\tab 83807.86\tab 1\tab 0\tab 1\tab 112542.58\tab 0\par
2\tab 3\tab 15619304\tab Onio\tab 502\tab France\tab Female\tab 42\tab 8\tab 159660.80\tab 3\tab 1\tab 0\tab 113931.57\tab 1\par
3\tab 4\tab 15701354\tab Boni\tab 699\tab France\tab Female\tab 39\tab 1\tab 0.00\tab 2\tab 0\tab 0\tab 93826.63\tab 0\par
4\tab 5\tab 15737888\tab Mitchell\tab 850\tab Spain\tab Female\tab 43\tab 2\tab 125510.82\tab 1\tab 1\tab 1\tab 79084.10\tab 0\par
df['HasCrCard'] = df['HasCrCard'].astype('category')\par
df['IsActiveMember'] = df['IsActiveMember'].astype('category')\par
df['Exited'] = df['Exited'].astype('category')\par
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])\par
df.head()\par
CreditScore\tab Geography\tab Gender\tab Age\tab Tenure\tab Balance\tab NumOfProducts\tab HasCrCard\tab IsActiveMember\tab EstimatedSalary\tab Exited\par
0\tab 619\tab France\tab Female\tab 42\tab 2\tab 0.00\tab 1\tab 1\tab 1\tab 101348.88\tab 1\par
1\tab 608\tab Spain\tab Female\tab 41\tab 1\tab 83807.86\tab 1\tab 0\tab 1\tab 112542.58\tab 0\par
2\tab 502\tab France\tab Female\tab 42\tab 8\tab 159660.80\tab 3\tab 1\tab 0\tab 113931.57\tab 1\par
3\tab 699\tab France\tab Female\tab 39\tab 1\tab 0.00\tab 2\tab 0\tab 0\tab 93826.63\tab 0\par
4\tab 850\tab Spain\tab Female\tab 43\tab 2\tab 125510.82\tab 1\tab 1\tab 1\tab 79084.10\tab 0\par
3. Perform Below Visualizations.\par
\u9679? Univariate Analysis\par
\u9679? Bi - Variate Analysis\par
\u9679? Multi - Variate Analysis\par
density = df['Exited'].value_counts(normalize=True).reset_index()\par
sns.barplot(data=density, x='index', y='Exited', );\par
density\par
index\tab Exited\par
0\tab 0\tab 0.7963\par
1\tab 1\tab 0.2037\par
\par
The data is significantly imbalanced\par
categorical = df.drop(columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'])\par
rows = int(np.ceil(categorical.shape[1] / 2)) - 1\par
\par
# create sub-plots anf title them\par
fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(10,6))\par
axes = axes.flatten()\par
\par
for row in range(rows):\par
    cols = min(2, categorical.shape[1] - row*2)\par
    for col in range(cols):\par
        col_name = categorical.columns[2 * row + col]\par
        ax = axes[row*2 + col]       \par
\par
        sns.countplot(data=categorical, x=col_name, hue="Exited", ax=ax);\par
        \par
plt.tight_layout()\par
\par
4. Perform descriptive statistics on the dataset.\par
df.info()\par
RangeIndex: 10000 entries, 0 to 9999\par
Data columns (total 11 columns):\par
 #   Column           Non-Null Count  Dtype   \par
---  ------           --------------  -----   \par
 0   CreditScore      10000 non-null  int64   \par
 1   Geography        10000 non-null  object  \par
 2   Gender           10000 non-null  object  \par
 3   Age              10000 non-null  int64   \par
 4   Tenure           10000 non-null  int64   \par
 5   Balance          10000 non-null  float64 \par
 6   NumOfProducts    10000 non-null  int64   \par
 7   HasCrCard        10000 non-null  category\par
 8   IsActiveMember   10000 non-null  category\par
 9   EstimatedSalary  10000 non-null  float64 \par
 10  Exited           10000 non-null  category\par
dtypes: category(3), float64(2), int64(4), object(2)\par
memory usage: 654.8+ KB\par
df.describe()\par
CreditScore\tab Age\tab Tenure\tab Balance\tab NumOfProducts\tab EstimatedSalary\par
count\tab 10000.000000\tab 10000.000000\tab 10000.000000\tab 10000.000000\tab 10000.000000\tab 10000.000000\par
mean\tab 650.528800\tab 38.921800\tab 5.012800\tab 76485.889288\tab 1.530200\tab 100090.239881\par
std\tab 96.653299\tab 10.487806\tab 2.892174\tab 62397.405202\tab 0.581654\tab 57510.492818\par
min\tab 350.000000\tab 18.000000\tab 0.000000\tab 0.000000\tab 1.000000\tab 11.580000\par
25%\tab 584.000000\tab 32.000000\tab 3.000000\tab 0.000000\tab 1.000000\tab 51002.110000\par
50%\tab 652.000000\tab 37.000000\tab 5.000000\tab 97198.540000\tab 1.000000\tab 100193.915000\par
75%\tab 718.000000\tab 44.000000\tab 7.000000\tab 127644.240000\tab 2.000000\tab 149388.247500\par
max\tab 850.000000\tab 92.000000\tab 10.000000\tab 250898.090000\tab 4.000000\tab 199992.480000\par
df.isna().sum()\par
CreditScore        0\par
Geography          0\par
Gender             0\par
Age                0\par
Tenure             0\par
Balance            0\par
NumOfProducts      0\par
HasCrCard          0\par
IsActiveMember     0\par
EstimatedSalary    0\par
Exited             0\par
dtype: int64\par
for i in df:\par
    if df[i].dtype=='object' or df[i].dtype=='category':\par
        print("unique of "+i+" is "+str(len(set(df[i])))+" they are "+str(set(df[i])))\par
unique of Geography is 3 they are \{'France', 'Germany', 'Spain'\}\par
unique of Gender is 2 they are \{'Male', 'Female'\}\par
unique of HasCrCard is 2 they are \{0, 1\}\par
unique of IsActiveMember is 2 they are \{0, 1\}\par
unique of Exited is 2 they are \{0, 1\}\par
def box_scatter(data, x, y):    \par
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16,6))\par
    sns.boxplot(data=data, x=x, ax=ax1)\par
    sns.scatterplot(data=data, x=x,y=y,ax=ax2)\par
box_scatter(df,'CreditScore','Exited');\par
plt.tight_layout()\par
print(f"# of Bivariate Outliers: \{len(df.loc[df['CreditScore'] < 400])\}")\par
# of Bivariate Outliers: 19\par
\par
box_scatter(df,'Age','Exited');\par
plt.tight_layout()\par
print(f"# of Bivariate Outliers: \{len(df.loc[df['Age'] > 87])\}")\par
# of Bivariate Outliers: 3\par
\par
box_scatter(df,'Balance','Exited');\par
plt.tight_layout()\par
print(f"# of Bivariate Outliers: \{len(df.loc[df['Balance'] > 220000])\}")\par
# of Bivariate Outliers: 4\par
\par
box_scatter(df,'EstimatedSalary','Exited');\par
plt.tight_layout()\par
\par
for i in df:\par
    if df[i].dtype=='int64' or df[i].dtypes=='float64':\par
        q1=df[i].quantile(0.25)\par
        q3=df[i].quantile(0.75)\par
        iqr=q3-q1\par
        upper=q3+1.5*iqr\par
        lower=q1-1.5*iqr\par
        df[i]=np.where(df[i] >upper, upper, df[i])\par
        df[i]=np.where(df[i] <lower, lower, df[i])\par
        \par
box_scatter(df,'CreditScore','Exited');\par
plt.tight_layout()\par
print(f"# of Bivariate Outliers: \{len(df.loc[df['CreditScore'] < 400])\}")\par
# of Bivariate Outliers: 19\par
\par
box_scatter(df,'Age','Exited');\par
plt.tight_layout()\par
print(f"# of Bivariate Outliers: \{len(df.loc[df['Age'] > 87])\}")\par
# of Bivariate Outliers: 0\par
\par
box_scatter(df,'Balance','Exited');\par
plt.tight_layout()\par
print(f"# of Bivariate Outliers: \{len(df.loc[df['Balance'] > 220000])\}")\par
# of Bivariate Outliers: 4\par
\par
from sklearn.preprocessing import LabelEncoder\par
encoder=LabelEncoder()\par
for i in df:\par
    if df[i].dtype=='object' or df[i].dtype=='category':\par
        df[i]=encoder.fit_transform(df[i])\par
x=df.iloc[:,:-1]\par
x.head()\par
CreditScore\tab Geography\tab Gender\tab Age\tab Tenure\tab Balance\tab NumOfProducts\tab HasCrCard\tab IsActiveMember\tab EstimatedSalary\par
0\tab 619.0\tab 0\tab 0\tab 42.0\tab 2.0\tab 0.00\tab 1.0\tab 1\tab 1\tab 101348.88\par
1\tab 608.0\tab 2\tab 0\tab 41.0\tab 1.0\tab 83807.86\tab 1.0\tab 0\tab 1\tab 112542.58\par
2\tab 502.0\tab 0\tab 0\tab 42.0\tab 8.0\tab 159660.80\tab 3.0\tab 1\tab 0\tab 113931.57\par
3\tab 699.0\tab 0\tab 0\tab 39.0\tab 1.0\tab 0.00\tab 2.0\tab 0\tab 0\tab 93826.63\par
4\tab 850.0\tab 2\tab 0\tab 43.0\tab 2.0\tab 125510.82\tab 1.0\tab 1\tab 1\tab 79084.10\par
y=df.iloc[:,-1]\par
y.head()\par
0    1\par
1    0\par
2    1\par
3    0\par
4    0\par
Name: Exited, dtype: int64\par
from sklearn.preprocessing import StandardScaler\par
scaler=StandardScaler()\par
x=scaler.fit_transform(x)\par
x\par
array([[-0.32687761, -0.90188624, -1.09598752, ...,  0.64609167,\par
         0.97024255,  0.02188649],\par
       [-0.44080365,  1.51506738, -1.09598752, ..., -1.54776799,\par
         0.97024255,  0.21653375],\par
       [-1.53863634, -0.90188624, -1.09598752, ...,  0.64609167,\par
        -1.03067011,  0.2406869 ],\par
       ...,\par
       [ 0.60524449, -0.90188624, -1.09598752, ..., -1.54776799,\par
         0.97024255, -1.00864308],\par
       [ 1.25772996,  0.30659057,  0.91241915, ...,  0.64609167,\par
        -1.03067011, -0.12523071],\par
       [ 1.4648682 , -0.90188624, -1.09598752, ...,  0.64609167,\par
        -1.03067011, -1.07636976]])\par
from sklearn.model_selection import train_test_split\par
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)\par
x_train.shape\par
(6700, 10)\par
x_test.shape\par
(3300, 10)\par
y_train.shape\par
(6700,)\par
y_test.shape\par
(3300,)\par
 \par
}
 
