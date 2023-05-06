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

#BEFORE TRANSFORMATION

![image](https://user-images.githubusercontent.com/119122478/236634943-1c87a117-ef7a-4e82-bf0d-2f7160714a60.png)

![image](https://user-images.githubusercontent.com/119122478/236634968-735881dc-d078-49df-a02f-ff00e44a90b2.png)

![image](https://user-images.githubusercontent.com/119122478/236634993-90468352-61d6-4500-87b7-ea4adaab4490.png)

![image](https://user-images.githubusercontent.com/119122478/236635015-c829cbf1-fdd9-4d73-9871-0fd35fd5a6a5.png)

![image](https://user-images.githubusercontent.com/119122478/236635031-5eebc473-624d-4311-bc4b-6be75b964022.png)


# RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully.






