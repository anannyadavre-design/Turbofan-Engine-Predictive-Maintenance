#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ==========================================
# NASA Turbofan Engine Predictive Maintenance
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



file_path = r"C:\Users\Anannya Davre\Downloads\train_FD001_prepared.csv"

df = pd.read_csv(file_path)

print("="*50)
print("DATASET LOADED SUCCESSFULLY")
print("="*50)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())



if 'unit_nr' in df.columns:
    engine_col = 'unit_nr'
elif 'engine_id' in df.columns:
    engine_col = 'engine_id'
else:
    raise Exception("Could not find engine ID column")



if 'RUL' not in df.columns:
    raise Exception("RUL column not found")



print("\nDataset Info:")
print(df.info())



engine1 = df[df[engine_col] == 1]

plt.figure(figsize=(12,5))
plt.plot(engine1['cycle'], engine1['RUL'])

plt.title("Engine 1 Remaining Useful Life")
plt.xlabel("Cycle")
plt.ylabel("RUL")
plt.grid()

plt.show()



sensor_cols = [c for c in df.columns if 'sensor' in c.lower()]

print("\nSensors Found:")
print(sensor_cols)



if len(sensor_cols) > 0:

    plt.figure(figsize=(12,5))

    plt.plot(
        engine1['cycle'],
        engine1[sensor_cols[0]]
    )

    plt.title(f"Engine 1 - {sensor_cols[0]}")
    plt.xlabel("Cycle")
    plt.ylabel("Sensor Value")
    plt.grid()

    plt.show()



drop_cols = [engine_col, 'cycle', 'RUL']

features = [c for c in df.columns if c not in drop_cols]

X = df[features]
y = df['RUL']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)



predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\n" + "="*50)
print("MODEL RESULTS")
print("="*50)

print("Mean Absolute Error:", round(mae,2), "cycles")


sample = X_test.iloc[[0]]

predicted_rul = model.predict(sample)[0]
actual_rul = y_test.iloc[0]

print("\nSample Prediction")
print("Predicted RUL:", round(predicted_rul,2))
print("Actual RUL:", actual_rul)


importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
    })

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features")

print(importance.head(10))



top10 = importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top10['Feature'],
    top10['Importance']
)

plt.title("Top 10 Important Sensors")
plt.xlabel("Importance")

plt.tight_layout()
plt.show()

print("\nAnalysis Complete.")


# In[ ]:




