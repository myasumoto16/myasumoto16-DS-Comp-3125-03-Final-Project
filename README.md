# Music and Mental Health Analysis

## 1. Introduction

- Objective: Explore connections between music preferences and mental health.
- Dataset: 'Music & Mental Health Survey Results' from Kaggle.
- **Key Questions**:
  1. What are the genre associated with the highest mental health issues?
  2. How much impact hours of music listening have on anxiety, depression, and insomnia levels?
  3. Can we accurately predict the anxiety, depression, and insomnia level based on their music taste or frequency using maching learning models?


## 2. Selection of Datasets

- DataSet: 'Music & Mental Health Survey Results' from Kaggle.
- https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results
- Data:
  - Collected between July 27, 2022, and November 8, 2022, via Google Form.
  - Publicly available under CC0 Public Domain license.
  - Three blocks:
    1. Musical background and preferences
    2. Listening habits
    3. Mental health indicators.
- Rationale: Despite discussions on self-report accuracy, the dataset aligns with this project's criteria.

## 3. Methodologies

### 3.1 Data Import and Cleaning

- Use **Pandas** to:
  - Filter dataset.
  - Apply `means()` method for average anxiety, depression, and insomnia levels.
- Use **Matplotlib** for visualization.
- Done for all questions

### 3.2 Machine Learning Models
- For Q3, "Can we accurately predict the axiety, depression, and insomnia level based on their music taste or frequency using maching learning models?"
- Utilize **sklearn models**:
  - DecisionTreeClassifier.
  - DecisionTreeRegressor.
- Dataset Division:
  - **Train** (80%)  and **test** (20%) subsets.
  - Input and output sets.
  -   Input: ['Fav genre'] or ['Hours per day']
  -   Output: ['Anxiety', 'Depression', 'Insomnia']
- Model Training:
  - Train subset used for model training with sklearn machine learning models
  - Test set evaluates prediction accuracy.
- Evaluation Metrics:
  - DecisionTreeClassifier: **accuracy score**. - 0 to 1
  - DecisionTreeRegressor: **mean squared error**.

**Objective:** Assess the practicality of using music preferences as predictors for understanding and predicting mental well-being.
