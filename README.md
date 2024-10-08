# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![Screenshot 2024-10-08 161652](https://github.com/user-attachments/assets/8723244a-236e-4934-94b7-3273689f1b27)

```
data.isnull().sum()
```
![Screenshot 2024-10-08 161657](https://github.com/user-attachments/assets/5bf94360-2c21-47ac-9638-1ab7ca6b74a3)

```

missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-10-08 161705](https://github.com/user-attachments/assets/09d7928d-c7b0-4777-8633-6816a03618ba)

```

data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-10-08 161712](https://github.com/user-attachments/assets/b63a5ec7-389a-4dcf-9322-87cde1c46a1f)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-10-08 161723](https://github.com/user-attachments/assets/0300b1c5-2526-416d-88eb-f1e7961e38f4)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-10-08 161731](https://github.com/user-attachments/assets/7f376d3a-3473-41b3-8201-ebd8fce0e101)

```
data
```
![Screenshot 2024-10-08 161737](https://github.com/user-attachments/assets/b100bf4c-18ba-4db4-b7a2-3d8e8c3a2cb5)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![Screenshot 2024-10-08 161746](https://github.com/user-attachments/assets/e482870e-b582-46dd-80f0-f089c74ec6fb)


```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-10-08 161811](https://github.com/user-attachments/assets/e4fcf453-7440-4331-be3a-6a3d17ea2b16)


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-10-08 161818](https://github.com/user-attachments/assets/355c1f09-bbd5-49ba-b262-0e7bc9ec1dd8)


```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-10-08 161822](https://github.com/user-attachments/assets/0d827813-f3bb-40d4-9151-0b202ed59ad8)


```
x=new_data[features].values
print(x)
```
![Screenshot 2024-10-08 161826](https://github.com/user-attachments/assets/84730e94-62f2-4486-85e0-9543ff4b0caf)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![Screenshot 2024-10-08 161829](https://github.com/user-attachments/assets/1a4a8673-dbb4-4d7f-88af-d13d500661d6)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![Screenshot 2024-10-08 161833](https://github.com/user-attachments/assets/cbe025c3-74c1-430c-a6a0-a1c18c374054)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![Screenshot 2024-10-08 161837](https://github.com/user-attachments/assets/63f15433-842d-493d-bfca-b048300c5d01)

```
data.shape
```
![Screenshot 2024-10-08 161841](https://github.com/user-attachments/assets/e8f30aa5-8d30-4fd4-8810-bf7c27ab6138)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-10-08 161851](https://github.com/user-attachments/assets/98f285ef-0d8f-47f3-b547-650fd96da0c6)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-10-08 161855](https://github.com/user-attachments/assets/36c14ca4-2378-458a-a7f8-d4ceeda4106d)

```
tips.time.unique()
```
![Screenshot 2024-10-08 161859](https://github.com/user-attachments/assets/d9472ef7-8d47-4fb8-96b4-44fe6dff8586)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-10-08 161903](https://github.com/user-attachments/assets/f0da14fb-bed1-46df-b922-bcfd4502b108)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![Screenshot 2024-10-08 161907](https://github.com/user-attachments/assets/af5fac67-fc4f-4d3e-a762-78e851422a63)

# RESULT:
  Thus, Feature selection and Feature scaling has been used on thegiven dataset.
