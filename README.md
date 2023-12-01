# Music and Mental Health Analysis

## 1. Introduction

- Objective: Explore connections between music preferences and mental health.
- Dataset: 'Music & Mental Health Survey Results' from Kaggle.
- Code: ([Music and Mental Health Analysis.ipynb](https://github.com/myasumoto16/myasumoto16-DS-Comp-3125-03-Final-Project/blob/main/Music%20and%20Mental%20Health%20Analysis.ipynb))
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
- For Q3, "Can we accurately predict the anxiety, depression, and insomnia level based on their music taste or frequency using maching learning models?"
- Utilize **sklearn models**:
  - **DecisionTreeRegressor**
  - a regression machine learning algorithm provided by sklearn, efficiently partitions input space, assigning mean values to regions. Its versatility in handling various data types, such as integers and strings makes it a powerful tool for accurate predictions for this project. 
- Dataset Division:
  - **Train** (90%)  and **test** (10%) subsets.
  - Input and output sets.
  -   Input: ['Fav genre'] or ['Hours per day']
  -   Output: ['Anxiety'], ['Depression'] or ['Insomnia']
- Model Training:
  - Train subset used for model training with sklearn machine learning models
  - Test set evaluates prediction accuracy.
- Evaluation Metrics:
  - DecisionTreeRegressor: **mean squared error**.
  - **mean squared error**: Squared value of the average value of error for prediction. 

**Objective:** Assess the practicality of using music preferences as predictors for mental well-being.

## 4. Result 
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
