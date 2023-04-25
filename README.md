# Ex-06-Feature-Transformation
AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature Transformation techniques to all the features of the data set

STEP 4
Save the data to the file


# CODE:

```

Name :NITHYAA SRI S.S
Register Number : 212222230100
Feature Transformation - Data_to_Transform.csv


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()

```

# Output

Feature Transformation - Data_to_Transform.csv

![image](https://user-images.githubusercontent.com/119122478/234178019-7e05c65b-ed99-4e46-bf50-1b84a93fe0ba.png)

![image](https://user-images.githubusercontent.com/119122478/234178099-c6dd8a0b-2bd0-46dc-89b0-858f6dda90a8.png)

![image](https://user-images.githubusercontent.com/119122478/234178176-b027859e-2f71-4924-9baa-4fa619075679.png)

![image](https://user-images.githubusercontent.com/119122478/234178247-311eaaf0-475b-4fc9-9cb7-001af41da8fe.png)

![image](https://user-images.githubusercontent.com/119122478/234178367-0bf71825-9b22-4366-8079-efa0b9963b21.png)

# Log Transformation

![image](https://user-images.githubusercontent.com/119122478/234178512-67e6e180-ed44-476c-824a-da68fc4702b1.png)

# Reciprocal Transformation


![image](https://user-images.githubusercontent.com/119122478/234178642-33922ef4-e794-4d56-87b5-3053c18980a0.png)

# SquareRoot Transformation

![image](https://user-images.githubusercontent.com/119122478/234178775-1c075ce2-09ae-4055-8192-bb7cb8d6b8da.png)

# Power Transformation

![image](https://user-images.githubusercontent.com/119122478/234178882-210574f4-a103-46f4-bab2-26fcd506b1de.png)

![image](https://user-images.githubusercontent.com/119122478/234178951-f79cc66e-2270-4d7c-b1f4-87d4281039c7.png)

# Quantile Transformation

![image](https://user-images.githubusercontent.com/119122478/234179043-01875620-2866-43d8-b4f1-2b4c9c5b56e8.png)

# RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully.






