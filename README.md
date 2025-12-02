# AISE-3000-Final-Project
# IMDb Movie Rating Prediction Project

## Overview
This project aims to build and evaluate machine learning models to predict IMDb movie ratings based on various movie attributes such as votes, meta score, director, tags, writers, and stars. The goal is to identify key factors influencing movie ratings and develop a robust predictive model.

## Table of Contents
1. [Project Objective](#project-objective)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results and Key Findings](#results-and-key-findings)
5. [Setup and Usage](#setup-and-usage)
6. [Dependencies](#dependencies)

## 1. Project Objective
The primary objective of this project is to:
- Analyze a dataset of IMDb movies to understand the distribution and relationships between various features and IMDb ratings.
- Preprocess and clean the data for machine learning tasks.
- Develop and evaluate several regression models to predict IMDb ratings.
- Identify the most influential features in predicting movie ratings.

## 2. Dataset
The dataset used is `imdb-top-rated-movies-user-rated.csv`, which contains information about top-rated movies, including:
- `Rank`: Movie rank.
- `Title`: Movie title.
- `IMDb Rating`: The target variable to be predicted.
- `Votes`: Number of votes the movie received.
- `Poster URL`, `Video URL`: URLs for media.
- `Meta Score`: Metacritic score.
- `Tags`: Genres/tags associated with the movie.
- `Director`: Movie director.
- `Description`: Movie synopsis.
- `Writers`: Writers of the movie.
- `Stars`: Main actors/stars of the movie.
- `Summary`: Brief summary.
- `Worldwide Gross`: Worldwide box office gross (sparse).

**Data Preprocessing Steps:**
- **Target Variable:** `IMDb Rating` was set as the target.
- **Feature Selection:** Key features chosen were `Votes`, `Meta Score`, `Director`, `Tags`, `Stars`, `Writers`.
- **`Votes` Cleaning:** The `Votes` column, initially in string format (e.g., '927K'), was converted to a numerical (float) type.
- **Missing Values:** Rows with missing `IMDb Rating` were dropped. Missing `Meta Score` and `Stars` were handled using imputation during the pipeline setup.

## 3. Methodology
The project followed these main steps:

### 3.1 Data Preparation
- Mounted Google Drive and loaded the `imdb-top-rated-movies-user-rated.csv` dataset.
- Initial data inspection (`df.info()`, `df.describe()`, `df.isna().sum()`).
- Defined target and features, and cleaned the `Votes` column.
- Split data into training (80%) and testing (20%) sets using `train_test_split` with `random_state=0`.
- Built a preprocessing pipeline using `ColumnTransformer`:
    - **Numerical Features** (`Votes`, `Meta Score`): Imputed missing values with the median and then scaled using `StandardScaler`.
    - **Categorical Features** (`Director`, `Tags`, `Stars`, `Writers`): Imputed missing values with the most frequent value and then applied `OneHotEncoder`.

### 3.2 Exploratory Data Analysis (EDA)
- **Correlation Matrix:** Computed and visualized the correlation between `Votes`, `Meta Score`, and `IMDb Rating` using a heatmap.
- **Histograms:** Visualized the distributions of `IMDb Rating`, `Votes`, and `Meta Score`.
- **Pair Plots:** Generated pair plots to examine relationships between numerical features.

### 3.3 Regression Modeling
- **Baseline Models:** Evaluated several regression models as baselines:
    - `DummyRegressor` (mean strategy)
    - `LinearRegression`
    - `Ridge Regression`
    - `RandomForestRegressor`
- Models were evaluated based on Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R^2) on the test set.
- **Hyperparameter Tuning:** `Ridge Regression`, identified as a strong baseline, was fine-tuned using `GridSearchCV` to optimize its `alpha` parameter with 5-fold cross-validation.

### 3.4 Model Evaluation and Interpretation
- **Tuned Model Performance:** Reported MAE, RMSE, and R^2 for the best-tuned Ridge model on the test set.
- **Feature Importance:** Extracted and analyzed the coefficients of the Ridge model to identify the most impactful features on IMDb ratings.
- **Prediction vs. Actual Plots:** Visualized predicted ratings against actual ratings to assess model fit.
- **Residual Analysis:** Plotted residuals against predicted values and examined the distribution of residuals to check for biases and model assumptions.

## 4. Results and Key Findings
- **Baseline Performance:** Ridge Regression typically performed well among the initial baseline models, showing a good balance between bias and variance.
- **Tuned Ridge Model:** The hyperparameter tuning confirmed that an `alpha` of 1.0 was optimal for the Ridge model, achieving competitive performance metrics (e.g., MAE of ~0.15, RMSE of ~0.19, R^2 of ~0.20 on the test set).
- **Feature Importance:** Specific directors, combinations of stars, and genre tags emerged as significant predictors of IMDb ratings, highlighting the qualitative aspects of movies as crucial factors.
- **Residuals:** The residual plots indicated a reasonable fit, with residuals generally scattered around zero, suggesting the model captures the underlying patterns effectively, though some variance remains unexplained.

## 5. Setup and Usage
To run this project, you will need a Google Colab environment or a local Python environment with the necessary dependencies.

### 5.1 Google Colab (Recommended)
1.  Open the `.ipynb` notebook in Google Colab.
2.  Mount your Google Drive to access the dataset. Ensure the `movies.zip` file is located at `/content/drive/MyDrive/DS3000_Project/movies.zip`.
3.  Run all cells sequentially.

### 5.2 Local Environment
1.  Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  Place the `movies.zip` file in the specified path or update the `zip_path` variable in the notebook.
3.  Install the required dependencies (see [Dependencies](#dependencies) section).
4.  Run the Jupyter Notebook cells.

## 6. Dependencies
This project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `zipfile` (built-in)

You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
