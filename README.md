# Student Performance Analysis & Final Grade Prediction System

A machine learning project that analyzes student academic, demographic, social, and behavioral characteristics and predicts final student performance.

## Project Overview

This project uses the UCI Student Performance dataset to explore patterns in student performance and build machine learning models for final grade prediction.

The project includes:

* Exploratory data analysis
* Data preprocessing
* Categorical variable encoding
* Random Forest regression
* TensorFlow neural network models
* Model performance comparison
* Feature importance analysis
* Interactive Streamlit application
* Data visualization
* Saved machine learning model

## Dataset

The project uses the **UCI Student Performance Dataset**.

The dataset contains information about **649 students** and includes academic, demographic, social, and behavioral characteristics.

The target variable is:

* `G3` — final grade, ranging from 0 to 20

`G1` and `G2` were excluded from the prediction features because they represent previous-period grades. Excluding them makes the model more suitable for predicting performance from student characteristics rather than simply using earlier grades to estimate the final grade.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow
* Streamlit
* Joblib
* UCI Machine Learning Repository

## Machine Learning Models

Three regression approaches were tested:

1. Random Forest
2. TensorFlow Neural Network
3. Improved TensorFlow Neural Network

### Model Results

| Model               | R² Score |
| ------------------- | -------: |
| Random Forest       |   0.1802 |
| TensorFlow          |   0.1082 |
| Improved TensorFlow |   0.0855 |

The Random Forest model achieved the highest R² score among the tested models.

For the Random Forest model:

* **MAE:** approximately 2.06 points
* **RMSE:** approximately 2.83 points
* **R²:** approximately 0.18

These results show that predicting final grades from the available characteristics is a challenging task. The model should therefore be considered an analytical prediction tool rather than a highly accurate grading system.

## Exploratory Data Analysis

The project includes visual analyses of:

* Final grade distribution
* Study time and final grade
* Previous failures and final grade
* Prediction errors
* Actual versus predicted grades
* Model performance comparison

The analyses are used to identify patterns and relationships in the dataset. They do not establish causal relationships between individual characteristics and student performance.

## Feature Importance

The Random Forest model identified several features as relatively important for its predictions.

Among the more important features were:

* Previous failures
* Absences
* School
* Father's education
* Alcohol consumption variables
* Free time
* Age
* Going out frequency
* Mother's education

Feature importance describes how much a feature contributes to the model's predictions. It does not mean that a feature directly causes higher or lower academic performance.

## Interactive Application

The project includes an interactive **Streamlit** application.

Users can enter student characteristics and receive a predicted final grade.

The application provides:

* Final grade prediction
* Model performance metrics
* Dataset visualizations
* Model comparison
* Explanations of the model
* Model limitations

## Project Structure

```text
student-performance-ai/
│
├── app.py
├── main.py
├── student_performance_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/elabenn/student-performance-ai.git
cd student-performance-ai
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in the browser at the local Streamlit address.

## Limitations

This project has several limitations:

* The dataset contains only 649 students.
* The model's predictive performance is limited.
* The test-set R² score is approximately 0.18 for the Random Forest model.
* The model should not be used to determine or assign real student grades.
* Feature importance should not be interpreted as causation.
* The model was developed and evaluated using a single train-test split.

## Future Improvements

Possible future improvements include:

* Testing additional machine learning algorithms
* Hyperparameter optimization
* Cross-validation
* More detailed error analysis
* Improving the Streamlit interface
* Adding model prediction confidence or uncertainty estimates
* Deploying the application online

## Project Goal

The main goal of this project is to demonstrate an end-to-end machine learning workflow, from data exploration and preprocessing to model training, evaluation, visualization, and deployment through an interactive application.