# Predictive Modeling of Employee Attrition  

This repository contains the final project for the Data Science Bootcamp at Dibimbing.id, focusing on building predictive models to analyze and mitigate employee attrition using data-driven techniques.  

## Table of Contents  
1. [Introduction](#introduction)  
2. [Dataset Overview](#dataset-overview)  
3. [Methodology](#methodology)  
4. [Key Findings](#key-findings)  
5. [Recommendations](#recommendations)  
6. [How to Run the Project](#how-to-run-the-project)  

## Introduction  
Employee attrition can significantly impact organizational stability, productivity, and costs. This project aims to:  
- Identify key factors influencing attrition using exploratory data analysis (EDA).  
- Develop predictive models using algorithms like XGBoost and Random Forest.  
- Provide actionable recommendations to improve employee retention.  

## Dataset Overview  
- **Source**: Kaggle Synthetic Employee Attrition Dataset  
- **Size**: 74,498 samples (training set: 35,767; testing set: 8,885)  
- **Features**: 16 relevant features after feature selection.  
- **Target Variable**: Attrition (Stayed or Left).  

## Methodology  
1. **Data Cleaning & Manipulation**:  
   - Removed outliers and unrealistic records.  
   - Applied scaling and encoding for categorical variables.  
2. **Exploratory Data Analysis (EDA)**:  
   - Analyzed correlations and distribution of features.  
   - Identified key features affecting attrition.  
3. **Feature Engineering**:  
   - Categorized monthly income into low, medium, and high.  
   - Selected 16 significant features for modeling.  
4. **Modeling**:  
   - Compared Random Forest and XGBoost.  
   - Optimized XGBoost with hyperparameter tuning (GridSearchCV).  
5. **Deployment**:  
   - Built an interactive web application using Streamlit.  

## Key Findings  
- **Top Features Influencing Attrition**:  
  - Job Level, Marital Status (Single), Remote Work, Work-Life Balance.  
- XGBoost achieved an F1-score of 0.76, showing balanced precision and recall.  

## Recommendations  
1. **Focus on Career Development**:  
   - Provide tailored career growth opportunities for employees at different levels.  
2. **Strengthen Remote Work Policies**:  
   - Offer flexible or hybrid work options to reduce attrition.  
3. **Improve Work-Life Balance**:  
   - Introduce employee wellness programs and flexible hours.  
4. **Engage Single Employees**:  
   - Design initiatives to support single employees who are at higher attrition risk.  

## How to Run the Project  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/your_username/attrition-prediction.git  
2. Install required libraries:
   ```bash
   pip install -r requirements.txt  
3.Launch the Streamlit application:
  ```bash
streamlit run final_app.py

Dataset: Kaggle Synthetic Employee Attrition
Demo Application: Streamlit Deployment
Contact:
LinkedIn: [linkedin.com/in/Alifgalabuana](https://www.linkedin.com/in/alifgalabuana/)

