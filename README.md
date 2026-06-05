# Turbofan-Engine-Predictive-Maintenance
Predictive maintenance of NASA turbofan engines using sensor data and Random Forest regression to estimate Remaining Useful Life (RUL).
# NASA Turbofan Engine Predictive Maintenance

## Project Overview

This project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset to predict the Remaining Useful Life (RUL) of aircraft engines using machine learning.

The workflow includes:

* Loading and exploring engine sensor data
* Visualizing engine degradation trends over operating cycles
* Analyzing sensor behavior
* Training a Random Forest Regression model
* Predicting Remaining Useful Life (RUL)
* Evaluating model performance using Mean Absolute Error (MAE)
* Identifying the most influential sensors through feature importance analysis

## Objective

Predict the number of operational cycles remaining before engine failure based on real-time sensor measurements. Accurate RUL prediction enables predictive maintenance, reduces unexpected downtime, and improves operational reliability.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

## Dataset

NASA C-MAPSS Turbofan Engine Dataset (FD001 subset)

Features include:

* Engine ID
* Operational Cycle
* Multiple Engine Sensor Readings
* Remaining Useful Life (RUL)

## Machine Learning Model

Random Forest Regressor

Model Parameters:

* 100 Decision Trees
* Random State = 42
* Parallel Processing Enabled (n_jobs = -1)

## Outputs

* RUL degradation curves
* Sensor trend visualizations
* Predicted vs Actual RUL values
* Mean Absolute Error (MAE)
* Feature importance ranking of sensors

## Applications

* Aerospace Maintenance
* Aircraft Health Monitoring
* Predictive Maintenance Systems
* Industrial IoT Analytics
* Digital Twin Development

## Future Improvements

* Hyperparameter optimization
* XGBoost and LightGBM comparison
* Deep learning approaches (LSTM)
* Real-time prediction dashboard
* Integration with a Digital Twin framework
