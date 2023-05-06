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
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()




```

# Output

Feature Transformation - Data_to_Transform.csv

![image](https://user-images.githubusercontent.com/119122478/236634316-68903489-2322-44f2-b500-7b861c33de89.png)

![image](https://user-images.githubusercontent.com/119122478/236634340-3d88369f-ca3d-4229-9118-8bde52bf2804.png)


![image](https://user-images.githubusercontent.com/119122478/236634372-c4fb030a-a34b-4636-91a4-a3e0c368ceda.png)

![image](https://user-images.githubusercontent.com/119122478/236634394-4f375b22-52b0-48f7-8b5a-6bdfcad94a6c.png)

![image](https://user-images.githubusercontent.com/119122478/236634419-6eaab676-d652-41e8-be77-fc3676648f6e.png)

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






