# Heart Disease Indicator Analysis

## Introduction

**Objective:** Explore possible indicators for heart diseases.

**Questions:**
1. Which age range are you most likely to have heart issues?
2. Do the smoking habits differ between people with and without heart issues?
3. Which machine learning model among LogisticRegression, XGBClassifier, KNeighborsClassifier is the most suitable for predicting heart issues?


## Selection of Datasets

**DataSet:** 'Heart Disease Health Indicators Dataset' from Kaggle.

**Data:**
- [Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)
    - This is a cleaned and filtered dataset that is specific to heart disease in 2015. 
    - 253,680 survey responses from cleaned BRFSS 2015 dataset.
    - Optimized for binary classifications for heart disease. '
    - Strong class imbalance. 
        - 229,787 people have not had heart disease while 23,893 have had heart disease.


- Original Dataset [Behavioral Risk Factor Surveillance System](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)
  - Public health surveys of more than 400,000 people from 2011 to 2015
   - Data on preventive health practices and behaviors that are linked to chronic diseases, injuries, and preventable infectious diseases in the adult population.
    - Collected by Center for Disease and Prevention

## Methodologies

### 1. Data Import and Cleaning

- Use Pandas to:
  - Import and filter dataset.
  - Apply `groupby()` method for grouping certain rows based on a value of another column 
- Use Matplotlib and seaborn for visualization.
  - Used to plot bar graphs and pie charts for Q1 and Q2.
  - Used to plot the confusion matrices for Q3. 

### 2. Machine Learning Models

- For Q3, "Which machine learning model among LogisticRegression, XGBClassifier, DecisionTreeClassifier is the most suitable for predicting heart issues?"
- Sklearn models below were picked based on their suitability for binary classifications:

#### 2.1. Logistic Regression

- Models the probability that a given instance belongs to a particular class.
- Ideal for binary classification tasks.
- Well-suited for linear relationships, provides interpretable coefficients, and outputs probabilities.
- Considerations: Assumes a linear relationship between features and log-odds, which might limit its ability to analyze non-linear patterns.
  
#### 2.2. XGBClassifier (XGBoost)

- Gradient boosting framework that builds a group of weak learners.
- Well-suited for binary classification tasks and often outperforms other algorithms.
- High predictive performance, handles non-linear relationships well, and includes regularization to prevent overfitting.
- Considerations: May require tuning of hyperparameters.

#### 2.3. DecisionTreeClassifier
- Supervised learning algorithm that constructs a tree structure to make predictions based on feature conditions.
- Well-suited for binary classification tasks due to its ability to create decision boundaries based on feature conditions.
- Assumes axis-aligned decision boundaries, potentially limiting its ability to capture complex, non-linear patterns.
- 
### 3. Dataset Division:

  - Train (90%) and test (10%) subsets.
  - Input and output sets.
    - Predicting the output ['HeartDiseaseorAttack'] based on the input set. 
    - Input: ['HighBP,	HighChol,	CholCheck,	BMI,	Smoker,	Stroke,	Diabetes,	PhysActivity,	HvyAlcoholConsump,	AnyHealthcare,	NoDocbcCost,	GenHlth	MentHlth,	PhysHlth,	DiffWalk,	Sex,	Age']
        - Columns used for the input set are chosen based on information that is commonly asked or measured at doctor's visits. 
        - Columns that are ignored are ['Fruits', 'Veggies', 'Education', 'Income']
    - Output: ['HeartDiseaseorAttack']
  
### 4. Handling Imbalance: 

- **SMOTE (Synthetic Minority Over-sampling Technique):**
    - addresses class imbalance in binary classification tasks.
    - designed for the minority class, SMOTE generates synthetic. 
    - By introducing synthetic examples, SMOTE helps balance class distribution, enhancing the model's ability to learn from the minority class and improving overall classification performance. 
        
### 5. Evaluation Metrics with report:

The classification report provides performance metrics for a binary classification model. Each row represents a class (0 or 1), and the columns include precision, recall, and F1-score. 

- **Precision:** The ratio of true positive predictions to the total predicted positives, indicating the accuracy of positive predictions.
- **Recall:** The ratio of true positive predictions to the total actual positives, measuring the model's ability to capture all positive instances.
- **F1-score:** The harmonic mean of precision and recall, offering a balanced assessment of a model's performance.


### 6. Visualization with Confusion Matrix

A confusion matrix is a table that provides a detailed summary of the performance of a classification model. It compares predicted labels against actual labels, categorizing instances into four outcomes:

- **True Positive (TP):** Instances correctly predicted as positive.
- **True Negative (TN):** Instances correctly predicted as negative.
- **False Positive (FP):** Instances incorrectly predicted as positive.
- **False Negative (FN):** Instances incorrectly predicted as negative.


## Results
- [Code](https://github.com/myasumoto16/myasumoto16-DS-Comp-3125-03-Final-Project/blob/main/Heart%20Disease.ipynb)
- **Disclaimer**: To run the code, go to Kernal -> Restart & Run All. It will take about 5 minutes for all the machine learning models to run 


### 1. Which age range are you most likely to have heart issues?
- The range of 60 - 64 has the most cases of heart diseases, followed by 75 - 79 and 65 - 69. 

<img src="figures/age_range.png" width="1000">

### 2. Do the smoking habits differ between people with and without heart issues?

#### 2.1. How many people with a heart disease smoke?

<img src="figures/smokers_disease.png" width="600">


#### 2.2. How many people without a heart disease smoke?

<img src="figures/smokers_non_disease.png" width="600">

- Compared to people without a heart disease, people with heart disease are more likely to be smokers. 

### 3. Can we accurately predict the anxiety, depression, and insomnia level based on their music taste or frequency using machine learning models?

#### 3.1. LogisticRegressor

**Accuracy:  0.7552822453484705 -> approximately 76%**
- The classification report below presents a detailed assessment of a binary classification model. While achieving high precision for the '0.0' class, the model shows lower recall, suggesting some instances are overlooked. On the other hand, for the '1.0' class, the model has higher recall but at the expense of precision, resulting in a trade-off between false positives and false negatives. The overall model accuracy is 76%.


<img src="figures/LogisticRegression.png" width="600">


  **Classification Report for LogisticRegressor:**

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 0.97      | 0.75   | 0.85     | 22971   |
| 1.0          | 0.25      | 0.80   | 0.38     | 2397    |
| **Accuracy** |           |        | 0.76     | 25368   |
| **Macro Avg**| 0.61      | 0.78   | 0.62     | 25368   |
| **Weighted Avg** | 0.91   | 0.76   | 0.80     | 25368   |




#### 3.2. XBGClassifier

**Accuracy:  0.8956953642384106 -> approximately 90%**

-  For the '0.0' class, the model demonstrates high precision (0.92) and recall (0.97), resulting in a F1-score of 0.94. However, for the '1.0' class, precision is lower (0.40) with a modest recall (0.21), leading to a F1-score of 0.28. The overall accuracy of the model is 90%.

<img src="figures/XGBClassifier.png" width="600">

**Classification Report for XBGClassifier:**  

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 0.92      | 0.97   | 0.94     | 22971   |
| 1.0          | 0.40      | 0.21   | 0.28     | 2397    |
| **Accuracy** |           |        | 0.90     | 25368   |
| **Macro Avg**| 0.66      | 0.59   | 0.61     | 25368   |
| **Weighted Avg** | 0.87   | 0.90   | 0.88     | 25368   |



#### 3.3. DecisionTreeClassifier
**Accuracy:  0.858483128350678 -> approximately 86%**
- The model achieves a high precision of 0.93 for the '0.0' class, with a high recall of 0.92. Additionally, the '1.0' class has a lower precision (0.27) but a low recall (0.29), leading to a F1-score of 0.28. The overall accuracy of the model is 86%. 

<img src="figures/DecisionTreeClassifier.png" width="600">

**Classification Report for DecisionTreeClassifier:**

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
|     0.0   |   0.93    |  0.92  |   0.92   |  22971  |
|     1.0   |   0.27    |  0.29  |   0.28   |   2397  |
|**Accuracy**|           |        |   0.86   |  25368  |
|**Macro Avg**|   0.60    |  0.60  |   0.60   |  25368  |
|**Weighted Avg**| 0.86  |  0.86  |   0.86   |  25368  |



## Discussion

### 1 Models

#### 1.1. Logistic Regression

It achieved an accuracy of approximately **76%**. The precision for the '0.0' class was high (0.97), 
-  accurate predictions of non-heart disease instances.
-  limitations in recall for the '1.0' class (0.80), suggesting it missed some instances of heart disease.

#### 1.2. XGBClassifier (XGBoost)

XGBClassifier outperformed Logistic Regression with an accuracy of approximately **90%**. 
- High precision (0.92) and recall (0.97) for the '0.0' class, 
- Robust predictions of non-heart disease instances.
- For the '1.0' class, the precision (0.40) and recall (0.21) were lower, suggesting room for improvement in identifying instances of heart disease.

#### 1.3 DecisionTreeClassifier
It achieved an accuracy of approximately **86%**. 
- High precision (0.93) and high recall (0.92) for the '0.0' class.
- For the '1.0' class, the precision (0.27) and recall (0.29) were lower compared to XGBClassifier.
  
### 2. Model Suitability

The choice of the most suitable model depends on the specific requirements and constraints of the problem at hand. In the context of predicting heart disease:

- **XGBClassifier (XGBoost)** appears to be the most accurate among the models considered. It excels in correctly predicting instances without heart disease but may benefit from further tuning to improve sensitivity to heart disease cases.
  
- **DecisionTreeClassifier** provides a balanced approach with moderate accuracy with 86%
  
- **Logistic Regression** could be suitable if interpretability is a priority, given its simplicity and the interpretability of its coefficients.

### 3. Further Steps

To enhance model performance, further steps can be taken:

- **Hyperparameter Tuning:** Fine-tuning hyperparameters, especially for XGBClassifier, may improve its performance.
  
- **Feature Engineering:** Exploring additional features or transforming existing ones might uncover patterns beneficial for prediction.

- **Explore Models:** Exploring additional models can be beneficial for more accurate prediction. 
 
- **Ensemble Methods:** Combining the predictions of multiple models through ensemble methods like stacking or boosting may lead to improved overall performance.


## Summary

In this heart disease indicator analysis, I analyzed age-related trends, differences in smoking habits, and evaluated three machine learning models: Logistic Regression, XGBClassifier (XGBoost), and KNeighborsClassifier for predicting heart issues.

### 1. Key Findings:

#### 1.1. **Age-Related Trends:**
   - Identified higher likelihoods of heart issues in specific age ranges (60 - 75), providing insights for targeted interventions.

#### 1.2. **Smoking Habits:**
   - Disparities in smoking habits between those with and without heart issues were observed, emphasizing the need for possible interventions for somkers. 

#### 1.3. **Model Performance:**
   - **XGBClassifier:** Outperformed with high accuracy (90%) in predicting non-heart disease instances, with room for improvement in identifying heart disease cases.

### 2. Further Steps:

- Explore hyperparameter tuning, feature engineering, additional models, and ensemble methods for enhanced predictive capabilities.

### 3. Implications for Healthcare:

- The results may inform targeted interventions and healthcare strategies, contributing to improved heart health outcomes.

## References
[1] [Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)  
[2] [CDC – 2015 BRFSS Survey Data and Documentation](https://www.cdc.gov/brfss/annual_data/annual_2015.html)  
[3] [scikit-learn Machine Learning in Python](https://scikit-learn.org/stable/)   
[4] [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)    





