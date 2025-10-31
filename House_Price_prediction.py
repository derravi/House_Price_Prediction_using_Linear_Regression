import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

#Load the Datasets
df = pd.read_csv("Data Sheet/house_data.csv")
print("Letse see the Data of this Datasets......\n")
df.head()

#Shape of the dataset.
print(f"The Total Row of this dataset is {df.shape[0]} and thetotal columns of this dataset is {df.shape[1]}.")

#Check the null values
print("Let see there is any null values or not........\n")
df.isnull().sum()

#Describe the full dataset.
print("Lets Descibe the data of the all columns........\n")
df.describe(include='all')

#Input and Output
print("Lets select the Feture.......\n")
fetures = ["Area_sqft","Bedrooms","Bathrooms","Floors","Year_Built"]

x = df[fetures]
y = df[["Price"]]

#Train Model 
model1 = LinearRegression()

model1.fit(x,y)
predicted_price = model1.predict(x)

#Velid Regression Matric
print("Lets see the Different parameters of this model.......\n")
mae = mean_absolute_error(y,predicted_price)
mse = mean_squared_error(y,predicted_price)
rmse = np.sqrt(mse)
r2_score = r2_score(y,predicted_price)

print("Mean Absolute Error(MAE):",round(mae,2))
print("Mean Squared Error(MSE):",round(mse,2))
print("Root Mean Squared Error(RMSE):",round(rmse,2))
print("R^2 Score:",round(r2_score,2))

with open("model/model.pkl","wb") as f:
    pickle.dump()

#Histogram Chart
print("Distribution of the House Price...........\n")
plt.figure(figsize=(10,6))
plt.hist(df["Price"],bins=50,color="red",edgecolor="black",label="Distribution of the House Price")
plt.title("Distribution of the House Price")
plt.xlabel("House Price")
plt.ylabel("Number of House")
plt.grid(True,color="skyblue")
plt.tight_layout()
plt.legend(loc='upper right')
plt.savefig("Diagram images/Distribution_of_the_House_Price.png",dpi=400,bbox_inches='tight')
plt.show()

#Scatter + Regression Line
print("House Area VS House Price.......\n")
plt.figure(figsize=(10,6))
plt.scatter(df[["Price"]],df[["Area_sqft"]],color="green",edgecolor="black",label="Actual Price")
plt.title("House Area VS House Price")
plt.xlabel("Price of Home")
plt.ylabel("Area in sqft")
plt.grid(True,color="skyblue")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("Diagram images/House_Area_VS_House_Price.png",dpi=400,bbox_inches='tight')
plt.show()

# Actual vs Predicted Prices Chart
print("Actual vs Predicted House Prices.......\n")
plt.figure(figsize=(10,6))
plt.scatter(y, predicted_price, color="green", edgecolor="black", alpha=0.6, label="Predicted vs Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="-", linewidth=2, label="Perfect Prediction Line")
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True, color="skyblue", linestyle="--", alpha=0.7)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("Diagram images/Actual_vs_Predicted_House_Prices.png", dpi=400, bbox_inches="tight")
plt.show()

# Predict the Price of the house based on the area in sqft

x = df[["Area_sqft","Bedrooms","Bathrooms","Floors","Year_Built"]]
y = df[["Price"]]

#Train Model 
model1 = LinearRegression()

model1.fit(x,y)

Area_sqft = float(input("Enter the Area of the house(in sqft):"))
bedrooms = int(input("Enter the number of Bedrooms:"))
bathrooms = float(input("Enter the Bathroom:"))
floors = float(input("Enter the number of the floors:"))
year_built = int(input("Enter the build year of the House:"))

predicted_price = model1.predict([[Area_sqft,bedrooms,bathrooms,floors,year_built]])

print(f"Area of House is {Area_sqft} and bedrooms is {bedrooms} and bathrooms is {bathrooms} in the {floors} floor and the build year is {year_built} so accordingly this all the infotmation the predicted price of the house is {predicted_price[0][0]:,.2f}.")