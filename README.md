# 🤖 AI Job Salary Prediction

## 📌 Project Overview

**AI Job Salary Prediction** is a Machine Learning regression project that predicts the salary of a job in USD based on job details, company information, experience, location, skills, and other job-related attributes.

The project helps demonstrate how Machine Learning can be used in recruitment analytics to understand salary patterns and support better salary and hiring decisions.

## 🎯 Objective

The main objective of this project is to build and compare multiple Machine Learning regression models and identify the model that provides the best salary predictions.

The target variable is:

**`salary_usd`** – Salary offered for the job in USD.

## 📊 Dataset

The dataset contains **15,000 records** with **18 input features and 1 target variable**.

Important features include:

* Job Title
* Experience Level
* Employment Type
* Company Location
* Company Size
* Company Name
* Employee Residence
* Education Required
* Years of Experience
* Required Skills
* Job Description Length
* Industry
* Remote Ratio
* Posting Date
* Application Deadline
* Benefits Score
* Salary Currency
* Job ID

## 🔄 Machine Learning Workflow

The project follows a simple Machine Learning pipeline:

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Feature Selection
     ↓
Hyperparameter Tuning
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Best Model Selection
```

### 1. Data Preprocessing

The data is prepared before training the models by:

* Handling missing values
* Scaling numerical features
* Encoding categorical features

### 2. Feature Selection

`SelectKBest` with `f_regression` is used to select the most important features for salary prediction.

### 3. Hyperparameter Tuning

`GridSearchCV` is used to find suitable hyperparameters for the Machine Learning models.

### 4. Models Used

The following regression models are trained and compared:

* **Linear Regression**
* **K-Nearest Neighbors (KNN) Regression**
* **Decision Tree Regression**
* **Random Forest Regression**

Random Forest is used as the **ensemble learning model**.

### 5. Model Evaluation

The models are evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

The model with the **lowest RMSE** is selected as the best-performing model.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* Matplotlib
* Machine Learning

## 📁 Project Structure

```text
AI-Job-Salary-Prediction/
│
├── salary_prediction.ipynb
├── ai_job_dataset.csv
├── README.md
└── requirements.txt
```

## 🚀 Future Deployment

The trained model can be deployed as a web application using **Streamlit**, allowing users to enter job details and receive an estimated salary.

## 💼 Business Impact

This project can help:

* Companies understand salary trends
* Recruiters make better salary decisions
* Organizations offer competitive salaries
* Job seekers understand salary expectations
* Hiring teams analyze factors affecting salary

## 👨‍💻 Conclusion

This project demonstrates an end-to-end Machine Learning workflow for a real-world regression problem. Multiple models are trained, tuned, and evaluated to determine which algorithm performs best for predicting job salaries.

The final objective is to select the model that provides the most accurate salary predictions based on the evaluation metrics.
