# Data Analysis Project: Sleep Quality Prediction using All of Us Dataset

## Project Overview
This project leverages the **All of Us** dataset to analyze sleep quality using Fitbit data, demographic, geographic, and lifestyle factors. The goal was to create a **machine learning model** that predicts sleep quality and provides actionable insights for improving sleep habits based on individual data.

## Key Highlights
- **Cohort Creation** based on Fitbit data and other lifestyle factors.
- Data **cleaning and preparation** using Python (Pandas, NumPy).
- **Machine learning model** development for predicting sleep quality.
- **Data visualization** of insights and model results using Matplotlib and Seaborn.
- **Web application suggestions** for deploying the model in a user-friendly platform.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Technologies Used](#technologies-used)
- [Data Preparation & Cleaning](#data-preparation--cleaning)
- [Machine Learning Model](#machine-learning-model)
- [Data Visualization](#data-visualization)
- [Web Application (Future Work)](#web-application-future-work)
- [Results](#results)

---

## Data
- The project uses the **All of Us dataset**, which includes data collected from a large cohort of individuals on various lifestyle, demographic, and health-related factors.
- **Data Types**:
  - **Fitbit Data**: Wearable data collected from participants (step count, sleep, etc.)
  - **Demographic Data**: Information such as age, gender, and ethnicity.
  - **Geographic Data**: Location information (e.g., zip code, region).
  - **Lifestyle Data**: Information on behaviors, habits, and health status.
  
The dataset is publicly available through the **All of Us Workbench** for use in health-related research.

---

## Technologies Used
- **Python**: Primary programming language for data analysis, model building, and visualization.
- **Pandas**: Data manipulation and cleaning.
- **NumPy**: Numerical computing for data preprocessing.
- **Jupyter Notebooks**: Environment used for data cleaning, machine learning modeling, and visualization.
- **Matplotlib**: For plotting and visualizing results.
- **Seaborn**: For creating more advanced visualizations.
- **scikit-learn**: For machine learning model development and evaluation.

---

## Data Preparation & Cleaning
- Data was accessed directly from the **All of Us Workbench**.
- The dataset was preprocessed using Python and the following steps:
  - **Filtering**: Selected individuals who have Fitbit data.
  - **Cleaning**: Handled missing values, outliers, and data imbalances.
  - **Normalization**: Scaled features as required for machine learning models.
  - **Feature Engineering**: Created new features based on existing data to improve model accuracy.

The code for data cleaning and preparation is located in `data_preprocessing.py`.

---

## Machine Learning Model
- **Model Training**:
  - The cleaned data was split into training and testing sets.
  - Several models were tested, including **Random Forest**, **Linear Regression**, and possibly others like **Logistic Regression** or **XGBoost**.
- **Evaluation**:
  - The models were evaluated based on **accuracy**, **precision**, **recall**, and **F1 score**.
  - Cross-validation was used to ensure the model's reliability.
  
The model training code is located in `model_training.ipynb`.

---

## Data Visualization
- Visualized the model results and insights using **Matplotlib** and **Seaborn**.
- Key visualizations include:
  - **Feature Importance**: Shows which features (e.g., step count, age, etc.) have the most impact on sleep quality.
  - **Correlation Matrix**: Displays correlations between various data features.
  - **Model Performance**: Accuracy and error metrics across different models.

The visualization code is located in `visualizations.py` and `model_results.ipynb`.

---

## Web Application (Future Work)
- **Web Application Plan**: 
  - A **Flask** or **Django** web app is suggested to deploy the trained model.
  - The web app would provide **real-time predictions** of sleep quality based on user input.
  - It would also offer **personalized recommendations** for improving sleep based on the user's data.

---

## Results
- The project achieved a prediction accuracy of **91%** using the **Random Forest model**.
- The model provided valuable insights into how factors like **physical activity**, **sleep patterns**, and **demographics** influence sleep quality.
- Future work would involve refining the model by adding more factors to understand more lifestyle dependancies and integrating it into a web application for real-world use.

---




