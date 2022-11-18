{\rtf1\ansi\ansicpg1252\deff0\deflang1033{\fonttbl{\f0\fnil\fcharset0 Calibri;}}
{\*\generator Msftedit 5.41.21.2510;}\viewkind4\uc1\pard\sa200\sl276\slmult1\lang9\f0\fs22\{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1033\{\\fonttbl\{\\f0\\fnil\\fcharset0 Calibri;\}\}\par
\{\\*\\generator Msftedit 5.41.21.2510;\}\\viewkind4\\uc1\\pard\\sa200\\sl276\\slmult1\\lang9\\f0\\fs22 1.Download the dataset\\par\par
2.Load the dataset\\par\par
import pandas as pd\\par\par
import numpy as np\\par\par
import seaborn as sns\\par\par
import matplotlib.pyplot as plt\\par\par
file=pd.read_csv("Churn_Modelling.csv")\\par\par
df=pd.DataFrame(file)\\par\par
df.head()\\par\par
RowNumber\\tab CustomerId\\tab Surname\\tab CreditScore\\tab Geography\\tab Gender\\tab Age\\tab Tenure\\tab Balance\\tab NumOfProducts\\tab HasCrCard\\tab IsActiveMember\\tab EstimatedSalary\\tab Exited\\par\par
0\\tab 1\\tab 15634602\\tab Hargrave\\tab 619\\tab France\\tab Female\\tab 42\\tab 2\\tab 0.00\\tab 1\\tab 1\\tab 1\\tab 101348.88\\tab 1\\par\par
1\\tab 2\\tab 15647311\\tab Hill\\tab 608\\tab Spain\\tab Female\\tab 41\\tab 1\\tab 83807.86\\tab 1\\tab 0\\tab 1\\tab 112542.58\\tab 0\\par\par
2\\tab 3\\tab 15619304\\tab Onio\\tab 502\\tab France\\tab Female\\tab 42\\tab 8\\tab 159660.80\\tab 3\\tab 1\\tab 0\\tab 113931.57\\tab 1\\par\par
3\\tab 4\\tab 15701354\\tab Boni\\tab 699\\tab France\\tab Female\\tab 39\\tab 1\\tab 0.00\\tab 2\\tab 0\\tab 0\\tab 93826.63\\tab 0\\par\par
4\\tab 5\\tab 15737888\\tab Mitchell\\tab 850\\tab Spain\\tab Female\\tab 43\\tab 2\\tab 125510.82\\tab 1\\tab 1\\tab 1\\tab 79084.10\\tab 0\\par\par
df['HasCrCard'] = df['HasCrCard'].astype('category')\\par\par
df['IsActiveMember'] = df['IsActiveMember'].astype('category')\\par\par
df['Exited'] = df['Exited'].astype('category')\\par\par
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])\\par\par
df.head()\\par\par
CreditScore\\tab Geography\\tab Gender\\tab Age\\tab Tenure\\tab Balance\\tab NumOfProducts\\tab HasCrCard\\tab IsActiveMember\\tab EstimatedSalary\\tab Exited\\par\par
0\\tab 619\\tab France\\tab Female\\tab 42\\tab 2\\tab 0.00\\tab 1\\tab 1\\tab 1\\tab 101348.88\\tab 1\\par\par
1\\tab 608\\tab Spain\\tab Female\\tab 41\\tab 1\\tab 83807.86\\tab 1\\tab 0\\tab 1\\tab 112542.58\\tab 0\\par\par
2\\tab 502\\tab France\\tab Female\\tab 42\\tab 8\\tab 159660.80\\tab 3\\tab 1\\tab 0\\tab 113931.57\\tab 1\\par\par
3\\tab 699\\tab France\\tab Female\\tab 39\\tab 1\\tab 0.00\\tab 2\\tab 0\\tab 0\\tab 93826.63\\tab 0\\par\par
4\\tab 850\\tab Spain\\tab Female\\tab 43\\tab 2\\tab 125510.82\\tab 1\\tab 1\\tab 1\\tab 79084.10\\tab 0\\par\par
3. Perform Below Visualizations.\\par\par
\\u9679? Univariate Analysis\\par\par
\\u9679? Bi - Variate Analysis\\par\par
\\u9679? Multi - Variate Analysis\\par\par
density = df['Exited'].value_counts(normalize=True).reset_index()\\par\par
sns.barplot(data=density, x='index', y='Exited', );\\par\par
density\\par\par
index\\tab Exited\\par\par
0\\tab 0\\tab 0.7963\\par\par
1\\tab 1\\tab 0.2037\\par\par
\\par\par
The data is significantly imbalanced\\par\par
categorical = df.drop(columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'])\\par\par
rows = int(np.ceil(categorical.shape[1] / 2)) - 1\\par\par
\\par\par
# create sub-plots anf title them\\par\par
fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(10,6))\\par\par
axes = axes.flatten()\\par\par
\\par\par
for row in range(rows):\\par\par
    cols = min(2, categorical.shape[1] - row*2)\\par\par
    for col in range(cols):\\par\par
        col_name = categorical.columns[2 * row + col]\\par\par
        ax = axes[row*2 + col]       \\par\par
\\par\par
        sns.countplot(data=categorical, x=col_name, hue="Exited", ax=ax);\\par\par
        \\par\par
plt.tight_layout()\\par\par
\\par\par
4. Perform descriptive statistics on the dataset.\\par\par
df.info()\\par\par
RangeIndex: 10000 entries, 0 to 9999\\par\par
Data columns (total 11 columns):\\par\par
 #   Column           Non-Null Count  Dtype   \\par\par
---  ------           --------------  -----   \\par\par
 0   CreditScore      10000 non-null  int64   \\par\par
 1   Geography        10000 non-null  object  \\par\par
 2   Gender           10000 non-null  object  \\par\par
 3   Age              10000 non-null  int64   \\par\par
 4   Tenure           10000 non-null  int64   \\par\par
 5   Balance          10000 non-null  float64 \\par\par
 6   NumOfProducts    10000 non-null  int64   \\par\par
 7   HasCrCard        10000 non-null  category\\par\par
 8   IsActiveMember   10000 non-null  category\\par\par
 9   EstimatedSalary  10000 non-null  float64 \\par\par
 10  Exited           10000 non-null  category\\par\par
dtypes: category(3), float64(2), int64(4), object(2)\\par\par
memory usage: 654.8+ KB\\par\par
df.describe()\\par\par
CreditScore\\tab Age\\tab Tenure\\tab Balance\\tab NumOfProducts\\tab EstimatedSalary\\par\par
count\\tab 10000.000000\\tab 10000.000000\\tab 10000.000000\\tab 10000.000000\\tab 10000.000000\\tab 10000.000000\\par\par
mean\\tab 650.528800\\tab 38.921800\\tab 5.012800\\tab 76485.889288\\tab 1.530200\\tab 100090.239881\\par\par
std\\tab 96.653299\\tab 10.487806\\tab 2.892174\\tab 62397.405202\\tab 0.581654\\tab 57510.492818\\par\par
min\\tab 350.000000\\tab 18.000000\\tab 0.000000\\tab 0.000000\\tab 1.000000\\tab 11.580000\\par\par
25%\\tab 584.000000\\tab 32.000000\\tab 3.000000\\tab 0.000000\\tab 1.000000\\tab 51002.110000\\par\par
50%\\tab 652.000000\\tab 37.000000\\tab 5.000000\\tab 97198.540000\\tab 1.000000\\tab 100193.915000\\par\par
75%\\tab 718.000000\\tab 44.000000\\tab 7.000000\\tab 127644.240000\\tab 2.000000\\tab 149388.247500\\par\par
max\\tab 850.000000\\tab 92.000000\\tab 10.000000\\tab 250898.090000\\tab 4.000000\\tab 199992.480000\\par\par
df.isna().sum()\\par\par
CreditScore        0\\par\par
Geography          0\\par\par
Gender             0\\par\par
Age                0\\par\par
Tenure             0\\par\par
Balance            0\\par\par
NumOfProducts      0\\par\par
HasCrCard          0\\par\par
IsActiveMember     0\\par\par
EstimatedSalary    0\\par\par
Exited             0\\par\par
dtype: int64\\par\par
for i in df:\\par\par
    if df[i].dtype=='object' or df[i].dtype=='category':\\par\par
        print("unique of "+i+" is "+str(len(set(df[i])))+" they are "+str(set(df[i])))\\par\par
unique of Geography is 3 they are \\\{'France', 'Germany', 'Spain'\\\}\\par\par
unique of Gender is 2 they are \\\{'Male', 'Female'\\\}\\par\par
unique of HasCrCard is 2 they are \\\{0, 1\\\}\\par\par
unique of IsActiveMember is 2 they are \\\{0, 1\\\}\\par\par
unique of Exited is 2 they are \\\{0, 1\\\}\\par\par
def box_scatter(data, x, y):    \\par\par
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16,6))\\par\par
    sns.boxplot(data=data, x=x, ax=ax1)\\par\par
    sns.scatterplot(data=data, x=x,y=y,ax=ax2)\\par\par
box_scatter(df,'CreditScore','Exited');\\par\par
plt.tight_layout()\\par\par
print(f"# of Bivariate Outliers: \\\{len(df.loc[df['CreditScore'] < 400])\\\}")\\par\par
# of Bivariate Outliers: 19\\par\par
\\par\par
box_scatter(df,'Age','Exited');\\par\par
plt.tight_layout()\\par\par
print(f"# of Bivariate Outliers: \\\{len(df.loc[df['Age'] > 87])\\\}")\\par\par
# of Bivariate Outliers: 3\\par\par
\\par\par
box_scatter(df,'Balance','Exited');\\par\par
plt.tight_layout()\\par\par
print(f"# of Bivariate Outliers: \\\{len(df.loc[df['Balance'] > 220000])\\\}")\\par\par
# of Bivariate Outliers: 4\\par\par
\\par\par
box_scatter(df,'EstimatedSalary','Exited');\\par\par
plt.tight_layout()\\par\par
\\par\par
for i in df:\\par\par
    if df[i].dtype=='int64' or df[i].dtypes=='float64':\\par\par
        q1=df[i].quantile(0.25)\\par\par
        q3=df[i].quantile(0.75)\\par\par
        iqr=q3-q1\\par\par
        upper=q3+1.5*iqr\\par\par
        lower=q1-1.5*iqr\\par\par
        df[i]=np.where(df[i] >upper, upper, df[i])\\par\par
        df[i]=np.where(df[i] <lower, lower, df[i])\\par\par
        \\par\par
box_scatter(df,'CreditScore','Exited');\\par\par
plt.tight_layout()\\par\par
print(f"# of Bivariate Outliers: \\\{len(df.loc[df['CreditScore'] < 400])\\\}")\\par\par
# of Bivariate Outliers: 19\\par\par
\\par\par
box_scatter(df,'Age','Exited');\\par\par
plt.tight_layout()\\par\par
print(f"# of Bivariate Outliers: \\\{len(df.loc[df['Age'] > 87])\\\}")\\par\par
# of Bivariate Outliers: 0\\par\par
\\par\par
box_scatter(df,'Balance','Exited');\\par\par
plt.tight_layout()\\par\par
print(f"# of Bivariate Outliers: \\\{len(df.loc[df['Balance'] > 220000])\\\}")\\par\par
# of Bivariate Outliers: 4\\par\par
\\par\par
from sklearn.preprocessing import LabelEncoder\\par\par
encoder=LabelEncoder()\\par\par
for i in df:\\par\par
    if df[i].dtype=='object' or df[i].dtype=='category':\\par\par
        df[i]=encoder.fit_transform(df[i])\\par\par
x=df.iloc[:,:-1]\\par\par
x.head()\\par\par
CreditScore\\tab Geography\\tab Gender\\tab Age\\tab Tenure\\tab Balance\\tab NumOfProducts\\tab HasCrCard\\tab IsActiveMember\\tab EstimatedSalary\\par\par
0\\tab 619.0\\tab 0\\tab 0\\tab 42.0\\tab 2.0\\tab 0.00\\tab 1.0\\tab 1\\tab 1\\tab 101348.88\\par\par
1\\tab 608.0\\tab 2\\tab 0\\tab 41.0\\tab 1.0\\tab 83807.86\\tab 1.0\\tab 0\\tab 1\\tab 112542.58\\par\par
2\\tab 502.0\\tab 0\\tab 0\\tab 42.0\\tab 8.0\\tab 159660.80\\tab 3.0\\tab 1\\tab 0\\tab 113931.57\\par\par
3\\tab 699.0\\tab 0\\tab 0\\tab 39.0\\tab 1.0\\tab 0.00\\tab 2.0\\tab 0\\tab 0\\tab 93826.63\\par\par
4\\tab 850.0\\tab 2\\tab 0\\tab 43.0\\tab 2.0\\tab 125510.82\\tab 1.0\\tab 1\\tab 1\\tab 79084.10\\par\par
y=df.iloc[:,-1]\\par\par
y.head()\\par\par
0    1\\par\par
1    0\\par\par
2    1\\par\par
3    0\\par\par
4    0\\par\par
Name: Exited, dtype: int64\\par\par
from sklearn.preprocessing import StandardScaler\\par\par
scaler=StandardScaler()\\par\par
x=scaler.fit_transform(x)\\par\par
x\\par\par
array([[-0.32687761, -0.90188624, -1.09598752, ...,  0.64609167,\\par\par
         0.97024255,  0.02188649],\\par\par
       [-0.44080365,  1.51506738, -1.09598752, ..., -1.54776799,\\par\par
         0.97024255,  0.21653375],\\par\par
       [-1.53863634, -0.90188624, -1.09598752, ...,  0.64609167,\\par\par
        -1.03067011,  0.2406869 ],\\par\par
       ...,\\par\par
       [ 0.60524449, -0.90188624, -1.09598752, ..., -1.54776799,\\par\par
         0.97024255, -1.00864308],\\par\par
       [ 1.25772996,  0.30659057,  0.91241915, ...,  0.64609167,\\par\par
        -1.03067011, -0.12523071],\\par\par
       [ 1.4648682 , -0.90188624, -1.09598752, ...,  0.64609167,\\par\par
        -1.03067011, -1.07636976]])\\par\par
from sklearn.model_selection import train_test_split\\par\par
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)\\par\par
x_train.shape\\par\par
(6700, 10)\\par\par
x_test.shape\\par\par
(3300, 10)\\par\par
y_train.shape\\par\par
(6700,)\\par\par
y_test.shape\\par\par
(3300,)\\par\par
 \\par\par
\}\par
}
 
