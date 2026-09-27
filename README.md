# Student Performance Analysis & Final Grade Prediction

A machine learning project that analyzes student academic and social characteristics and predicts final student performance.

## Project Overview

This project uses the UCI Student Performance dataset to explore factors associated with students' final grades and build machine learning models for final grade prediction.

The project includes:

- Exploratory data analysis
- Data preprocessing
- Categorical variable encoding
- Random Forest regression
- TensorFlow neural network models
- Model performance comparison
- Interactive Streamlit application
- Data visualization
- Saved machine learning model

## Dataset

The project uses the **UCI Student Performance Dataset**.

The dataset contains information about **649 students** and includes academic, demographic, social, and behavioral characteristics.

The target variable is:

- `G3` — final grade, ranging from 0 to 20

`G1` and `G2` were excluded from the prediction features because they are previous-period grades and could make the final-grade prediction less meaningful as a general student-profile model.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Streamlit
- Joblib
- UCI Machine Learning Repository

## Machine Learning Models

Three regression approaches were tested:

1. Random Forest
2. TensorFlow Neural Network
3. Improved TensorFlow Neural Network

### Model Results

| Model | R² Score |
|---|---:|
| Random Forest | 0.1802 |
| TensorFlow | 0.1082 |
| Improved TensorFlow | 0.0855 |

The Random Forest model achieved the highest R² score among the tested models.

For the Random Forest model:

- MAE: approximately 2.06 points
- RMSE: approximately 2.83 points
- R²: approximately 0.18

The results show that predicting student final grades from the available characteristics is a challenging task. The model should therefore be considered an analytical prediction tool rather than a highly accurate grading system.

## Exploratory Data Analysis

The project includes several visual analyses:

- Distribution of final grades
- Study time and final grade relationship
- Number of previous failures and final grade relationship
- Model performance comparison

These visualizations are used to explore patterns in the dataset rather than to establish causal relationships.

## Feature Importance

The Random Forest model identified several features as relatively important for its predictions.

Some of the most important features included:

- Previous failures
- Absences
- School
- Father's education
- Alcohol consumption variables
- Free time
- Age
- Going out frequency
- Mother's education

Feature importance indicates how much a feature contributes to the model's predictions. It does not demonstrate that a feature directly causes a student's performance.

## Interactive Application

The project includes an interactive Streamlit application.

Users can enter student characteristics and receive a predicted final grade.

The application also provides:

- Prediction results
- Model performance metrics
- Dataset visualizations
- Model comparison
- Explanations of the model and its limitations

## Project Structure

```text
student-performance-ai/
│
├── app.py
├── main.py
├── student_performance_model.pkl
├── requirements.txt
├── README.md
└── .venv/