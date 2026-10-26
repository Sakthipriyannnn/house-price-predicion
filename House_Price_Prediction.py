Python 3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import numpy as np
import pandas as pd
... import matplotlib.pyplot as plt
... from sklearn.model_selection import train_test_split
... from sklearn.linear_model import LinearRegression
... from sklearn.metrics import mean_absolute_error, r2_score
... data={"area_sqft":[800,1000,1200,1500,1800,2000,2200,2400,2600,2800],
...       "bedrooms":[2,2,3,3,3,3,4,4,4,5],
...       "age_years":[10,8,5,4,2,8,5,3,2,1],
...       "price_in_lakhs":[15,20,25,32,40,42,48,55,60,68]}
... df=pd.DataFrame(data)
... plt.scatter(df["area_sqft"], df["price_in_lakhs"])
... plt.xlabel("Area (SqFt)")
... plt.ylabel("Price (lakh)")
... plt.title("Distribution of House price")
... plt.show()
... print(df.corr().round(2))
... #check missing values
... print("\nMissing values:")
... print(df.isnull().sum())
... #check duplicate rows
... print("\nDuplicate rows:")
... print(df.duplicated())
... #check data type
... print("\nData Type:")
... print(df.dtypes)
... #check dataset size
... print("\nDataset shape:",df.shape)
... X=df[["area_sqft", "bedrooms","age_years"]]
... y=df["price_in_lakhs"]
... print("\nInput Features:")
... print(X.head())
... print("\nTarget Price:")
... print(y.head())
... X_train, X_test, y_train,y_test=train_test_split(X,yop
...                                                  ,test_size=0.2, random_state=42)
... print("\nTraining input shape:",X_train.shape)
... print("Testing input shape:",X_test.shape)
... print("\nTraining target shape:",y_train.shape)
... print("Testing target shape:",y_test.shape)
... model=LinearRegression()
... model.fit(X_train, y_train)
... print("\nModel training Complete:")
... print("Coefficients:",model.coef_)
... print("Intercept:",model.intercept_)
... y_pred=model.predict(X_test)
... print("\nActual Price:",y_test.values)
... print("Predicted prices:", y_pred)
... #Calculate MAE
mae=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error:",mae)
#Calculate R_Square score
r2=r2_score(y_test,y_pred)
print("R2 Score:",r2)
#visualize vs predicted price
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()])
plt.xlabel("Actual Price (lakhs)")
plt.ylabel("Predicted price (lakhs)")
plt.show()
#predict the new house price
new_h=pd.DataFrame({"area_sqft":[1600],
                    "bedrooms":[3],
                    "age_years":[4]})
predict_price=model.predict(new_h)
print("\nPredicted new House Price:",predict_price,"lakhs")
print("-------------------------------------------------")
#numpy
np.random.seed(42)
n=200
area_sqft=np.random.randint(600,3000,n)
bedrooms=np.random.randint(1,6,n)
age_years=np.random.randint(0,31,n)
#simulated price in lakhs
price_lakhs =(area_sqft*0.025 + bedrooms*2 - age_years*0.3 + np.random.normal(0,3,n))
price_lakhs = np.maximum(price_lakhs,5)
df=pd.DataFrame({"area_sqft":area_sqft,
                 "bedrooms":bedrooms,
                 "age_years":age_years,
                 "price_lakhs":price_lakhs.round(2)})
print(df.head())
print("\nDataset shape:",df.shape)
X=df[["area_sqft","bedrooms","age_years"]]
y=df["price_lakhs"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
print("Training rows:",len(X_train))
print("Testing rows:",len(X_train))
model=LinearRegression()
model.fit(X_train,y_train)
print("Model training completed!")
#prediction
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("\nMean Absolute Error:",round(mae,2),"lakhs")
print("r2_score:",round(r2,2),"%")
#Visualize
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()])
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (lakhs)")
plt.title("Actual Price V Predicted Price")
plt.show()

