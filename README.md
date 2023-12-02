# Heart Disease Indicator Analysis

## 1. Introduction

**Objective:** Explore possible indicators for heart diseases.

**Questions:**
1. Which age range are you most likely to have heart issues?
2. Do the smoking habits differ between people with and without heart issues?
3. Which machine learning model among LogisticRegression, RandomForestClassifier, XGBClassifier, KNeighborsClassifier is the most suitable for predicting heart issues?


## 2. Selection of Datasets

**DataSet:** 'Heart Disease Health Indicators Dataset' from Kaggle.

**Data:**
- [Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)
    - This is a cleaned and filtered dataset that is specific to heart disease in 2015. 
    - 253,680 survey responses from cleaned BRFSS 2015 dataset.
    - Optimized for binary classifications for heart disease. '
    - Strong class imbalance. 
        - 229,787 people have not had heart disease while 23,893 have had heart disease.


- Original Dataset [Behavioral Risk Factor Surveillance System](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)
    - Collected by Center for Disease and Prevention

## 3. Methodologies

### 3.1. Data Import and Cleaning

- Use Pandas to:
  - Filter dataset.
  - Apply `groupby()` method for grouping certain rows based on a value of another column 
- Use Matplotlib and seaborn for visualization.

### 3.2. Machine Learning Models

- For Q3, "Which machine learning model among LogisticRegression, RandomForestClassifier, XGBClassifier, KNeighborsClassifier is the most suitable for predicting heart issues?"
- Sklearn models below were picked based on their suitability for binary classifications:

#### 3.2.1. Logistic Regression

- Nature: Logistic Regression models the probability that a given instance belongs to a particular class.
- Use Case for Binary: Ideal for binary classification tasks.
- Strengths for Binary: Well-suited for linear relationships, provides interpretable coefficients, and outputs probabilities.
- Considerations: Assumes a linear relationship between features and log-odds, which might limit its ability to analyze non-linear patterns.

#### 3.2.2. RandomForestClassifier

- Nature: RandomForestClassifier is a group of decision trees that work together to make predictions.
- Use Case for Binary: Effective for binary classification tasks due to its ability to capture complex non-linear relationships.
- Strengths for Binary: Robust, handles feature interactions well, and can deal with imbalanced datasets.
- Considerations: Computationally intensive for large datasets, and the interpretability of the individual trees may be limited.

#### 3.2.3. XGBClassifier (XGBoost)

- Nature: Gradient boosting framework that builds a group of weak learners.
- Use Case for Binary: Well-suited for binary classification tasks and often outperforms other algorithms.
- Strengths for Binary: High predictive performance, handles non-linear relationships well, and includes regularization to prevent overfitting.
- Considerations: May require tuning of hyperparameters.

#### 3.2.4. KNeighborsClassifier

- Nature: Classifications of instances based on the majority class of their k-nearest neighbors.
- Use Case for Binary: Suitable for binary classification tasks, especially when local patterns in the data are important.
- Strengths for Binary: Simple conceptually, non-linear relationships are implicitly captured.
- Considerations: Sensitive to the choice of k, computationally expensive for large datasets, and may not perform well in high-dimensional spaces.

### 3.3. Dataset Division:

  - Train (90%) and test (10%) subsets.
  - Input and output sets.
    - Input: ['HighBP,	HighChol,	CholCheck,	BMI,	Smoker,	Stroke,	Diabetes,	PhysActivity,	HvyAlcoholConsump,	AnyHealthcare,	NoDocbcCost,	GenHlth	MentHlth,	PhysHlth,	DiffWalk,	Sex,	Age']
        - Columns used for the input set are chosen based on information that is commonly asked or measured at doctor's visits. 
        - Columns that are ignored are ['Fruits', 'Veggies', 'Education', 'Income']
    - Output: ['HeartDiseaseorAttack']
  
### 3.4. Handling Imbalance: 

- **SMOTE (Synthetic Minority Over-sampling Technique):** 
    - addresses class imbalance in binary classification tasks.
    - designed for the minority class, SMOTE generates synthetic. 
    - By introducing synthetic examples, SMOTE helps balance class distribution, enhancing the model's ability to learn from the minority class and improving overall classification performance. 
        
### 3.5. Evaluation Metrics with report:

The classification report provides performance metrics for a binary classification model. Each row represents a class (0 or 1), and the columns include precision, recall, and F1-score. 

- **Precision:** The ratio of true positive predictions to the total predicted positives, indicating the accuracy of positive predictions.
- **Recall:** The ratio of true positive predictions to the total actual positives, measuring the model's ability to capture all positive instances.
- **F1-score:** The harmonic mean of precision and recall, offering a balanced assessment of a model's performance.


### 3.6. Visualization with Confusion Matrix

A confusion matrix is a table that provides a detailed summary of the performance of a classification model. It compares predicted labels against actual labels, categorizing instances into four outcomes:

- **True Positive (TP):** Instances correctly predicted as positive.
- **True Negative (TN):** Instances correctly predicted as negative.
- **False Positive (FP):** Instances incorrectly predicted as positive.
- **False Negative (FN):** Instances incorrectly predicted as negative.


## 4. Results
### Q1: What are the genres associated with the highest mental health issues?
**1. What are the genres associated with the highest anxiety levels?**

| Rank | Genre               | Anxiety Level     |
|------|---------------------|-----------|
| 1    | **Folk**                | 6.566667  |
| 2    | K-pop               | 6.230769  |
| 3    | Hip hop             | 6.200000  |
| 4    | Rock                | 6.122340  |
| 5    | Lofi                | 6.100000  |

![Alt text](figures/anxiety_genre.png?raw=true "anxiety_genre")

**2. What are the genres associated with the highest depression levels?**

| Rank | Genre               | Depression Level     |
|------|---------------------|-----------|
| 1    | **Lofi**                | 6.600000  |
| 2    | Hip hop             | 5.800000  |
| 3    | EDM                 | 5.243243  |
| 4    | Rock                | 5.236702  |
| 5    | Metal               | 5.068182  |


![Alt text](figures/depression_genre.png?raw=true "depression_genre")

**3. What are the genres associated with the highest insomnia levels?**

| Rank | Genre               | Insomnia Level     |
|------|---------------------|-----------|
| 1    | **Lofi**                | 5.600000  |
| 2    | Gospel              | 5.333333  |
| 3    | Metal               | 4.556818  |
| 4    | Video game music    | 4.000000  |
| 5    | EDM                 | 3.972973  |


![Alt text](figures/insomnia_genre.png?raw=true "insomnia_genre")

### Q2: How much impact hours of music listening have on anxiety, depression, and insomnia levels?
**1. How much impact hours of music listening have on anxiety level?**

| Rank | Hours per day  | Anxiety Level    |
|------|----------|------------|
| 1    | 13.0     | 10.000000  |
| 2    | 9.0      | 9.666667   |
| 3    | 18.0     | 9.000000   |
| 4    | 20.0     | 8.000000   |
| 5    | 10.0     | 6.700000   |


![Alt text](figures/anxiety_hours.png?raw=true "anxiety_hours")

**2. How much impact hours of music listening have on depression level?**

| Rank | Hours per day  | Depression Level    |
|------|----------|------------|
| 1    | 13.0     | 10.000000  |
| 2    | 20.0     | 10.000000  |
| 3    | 14.0     | 10.000000  |
| 4    | 18.0     | 8.000000   |
| 5    | 12.0     | 7.111111   |


![Alt text](figures/depression_hours.png?raw=true "depression_hours")

**3. How much impact hours of music listening have on insomnia level?**

| Rank | Hours per day | Insomnia Level    |
|------|----------|------------|
| 1    | 13.0     | 10.000000  |
| 2    | 20.0     | 10.000000  |
| 3    | 14.0     | 8.000000   |
| 4    | 9.0      | 7.000000   |
| 5    | 15.0     | 6.000000   |

![Alt text](figures/insomnia_hours.png?raw=true "insomnia_hours")

### Q3: Can we accurately predict the anxiety, depression, and insomnia level based on their music taste or frequency using machine learning models?

**1. Can we accurately predict the anxiety level based on hours per day spent on music listening?**

**Mean Squared Error (MSE):  7.927178846908276.**
- Since the MSE is about 7.927, the average value of error in prediction of anxiety level is approximately **2.81**. For example, if the model predicts an anxiety level of 6, the actual value could be between 3.19 and 8.81

![Alt text](figures/predict_anxiety_hours.png?raw=true "predict_anxiety_hours")

- In the figure above, we expect to see a linear relationship between actual and predicted anxiety values, which means as the predicted values increase, the actual values would increase. However, As seen in the figure, most of the predicted anxiety values are between 5 and 7 regardless of their corresponding actual anxiety levels. This observation aligns with the MSE of 7.927 as well since it's relatively large considering that the anxiety level collected is between 0 and 10. 

**2. Can we accurately predict the depression level based on hours per day spent on music listening?**

**Mean Squared Error:  9.884870854441782**
- Since the MSE is about 9.884, the average value of error in prediction of anxiety level is approximately **3.14**. For example, if the model predicts a depression level of 6, the actual value could be between 2.86 and 9.14

![Alt text](figures/predict_depression_hours.png?raw=true "predict_depression_hours")

- In the figure above, we expect to see a linear relationship between actual and predicted depression values. However, most of the predicted depression values are between 4 and 7 regardless of their corresponding actual depression levels. This observation also aligns with the MSE of 9.884. 

**3. Can we accurately predict the insomnia level based on hours per day spent on music listening?**
**Mean Squared Error:  10.33583624994101**
- Since the MSE is about 10.335, the average value of error in prediction of anxiety level is approximately **3.21**. For example, if the model predicts an insomnia level of 6, the actual value could be between 2.79 and 9.21.
  
![Alt text](figures/predict_insomnia_hours.png?raw=true "predict_insomnia_hours")

- In the figure above, we expect to see a linear relationship between actual and predicted insomnia levels. However, most of the predicted insomnia levels are between 3 and 7 regardless of their corresponding actual insomnia levels. There is no linear progression in the graph. This observation also aligns with the MSE of 10.335. 

**4. Can we accurately predict the anxiety level based on their music taste?**

**Mean Squared Error:  8.056713371052219**
- Since the MSE is about 8.056, the average value of error in prediction of anxiety level is approximately **2.83**. For example, if the model predicts an anxiety level of 6, the actual value could be between 3.17 and 8.83.

![Alt text](figures/predict_anxiety_genre.png?raw=true "predict_anxiety_genre")

- In the figure above, we expect to see a linear relationship between actual and predicted anxiety levels. However, all of the predicted anxiety levels are between 4.8 and 6.4 regardless of their corresponding actual anxiety levels. There is no linear relationships observed in the graph. This observation also aligns with the MSE of 9.884. 

**5. Can we accurately predict the depression level based on their music taste?**

**Mean Squared Error:  9.107010534791192**

- Since the MSE is about 9.107, the average value of error in prediction of anxiety level is approximately **3.01**. For example, if the model predicts a depression level of 6, the actual value could be between 2.99 and 9.01


![Alt text](figures/predict_depression_genre.png?raw=true "predict_depression_genre")

**6. Can we accurately predict the insomnia level based on their music taste?**

**Mean Squared Error:  9.782376453165558**

- Since the MSE is about 10.335, the average value of error in prediction of anxiety level is approximately **3.12**. For example, if the model predicts an insomnia level of 6, the actual value could be between 2.88 and 9.12.
  
![Alt text](figures/predict_insomnia_genre.png?raw=true "predict_insomnia_genre")
