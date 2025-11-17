import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

# ==============================
# Create required directories
# ==============================
os.makedirs("model", exist_ok=True)
os.makedirs("Diagram images", exist_ok=True)

# ==============================
# Load the Dataset
# ==============================
df = pd.read_csv("Data Sheet/house_data.csv")
print("Let's see the data of this dataset......\n")
print(df.head())

# Shape of the dataset.
print(
    f"The total rows of this dataset are {df.shape[0]} "
    f"and the total columns of this dataset are {df.shape[1]}."
)

# Check the null values
print("\nLet's see if there are any null values or not........\n")
print(df.isnull().sum())

# Describe the full dataset.
print("\nLet's describe the data of all columns........\n")
print(df.describe(include='all'))

# ==============================
# Input and Output
# ==============================
print("\nLet's select the features.......\n")
features = ["Area_sqft", "Bedrooms", "Bathrooms", "Floors", "Year_Built"]

X = df[features]
y = df["Price"]  # 1D Series is better for sklearn

# ==============================
# Train Model
# ==============================
model1 = LinearRegression()
model1.fit(X, y)

predicted_price = model1.predict(X)

# ==============================
# Validate Regression Metrics
# ==============================
print("\nLet's see the different parameters of this model.......\n")

y_true = y.values  # 1D array
y_pred = predicted_price  # 1D array

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R^2 Score:", round(r2, 2))

# ==============================
# Save Model with Pickle
# ==============================
with open("model/model.pkl", "wb") as f:
    pickle.dump(model1, f)

# ==============================
# Histogram Chart
# ==============================
print("\nDistribution of the House Price...........\n")
plt.figure(figsize=(10, 6))
plt.hist(df["Price"], bins=50, color="red", edgecolor="black",
         label="Distribution of the House Price")
plt.title("Distribution of the House Price")
plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.grid(True, color="skyblue")
plt.tight_layout()
plt.legend(loc='upper right')
plt.savefig("Diagram images/Distribution_of_the_House_Price.png",
            dpi=400, bbox_inches='tight')
plt.show()

# ==============================
# Scatter: House Area VS House Price
# ==============================
print("House Area VS House Price.......\n")
plt.figure(figsize=(10, 6))
plt.scatter(df["Area_sqft"], df["Price"],
            color="green", edgecolor="black", label="Actual Price")
plt.title("House Area VS House Price")
plt.xlabel("Area in sqft")
plt.ylabel("Price of House")
plt.grid(True, color="skyblue")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("Diagram images/House_Area_VS_House_Price.png",
            dpi=400, bbox_inches='tight')
plt.show()

# ==============================
# Actual vs Predicted Prices Chart
# ==============================
print("Actual vs Predicted House Prices.......\n")
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, color="green", edgecolor="black",
            alpha=0.6, label="Predicted vs Actual")
plt.plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()],
    color="red", linestyle="-", linewidth=2,
    label="Perfect Prediction Line"
)
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True, color="skyblue", linestyle="--", alpha=0.7)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("Diagram images/Actual_vs_Predicted_House_Prices.png",
            dpi=400, bbox_inches="tight")
plt.show()

# ==============================
# Predict the Price of a House from User Input
# ==============================
print("\nNow let's predict the price of a house based on your input...\n")

Area_sqft = float(input("Enter the Area of the house (in sqft): "))
bedrooms = int(input("Enter the number of Bedrooms: "))
bathrooms = float(input("Enter the number of Bathrooms: "))
floors = float(input("Enter the number of Floors: "))
year_built = int(input("Enter the build year of the House: "))

user_features = [[Area_sqft, bedrooms, bathrooms, floors, year_built]]
predicted_price_input = model1.predict(user_features)

print(
    f"\nArea of house is {Area_sqft} sqft, bedrooms: {bedrooms}, "
    f"bathrooms: {bathrooms}, floors: {floors}, built in {year_built}."
)
print(
    f"According to this information, the predicted price of the house is "
    f"{predicted_price_input[0]}."
)