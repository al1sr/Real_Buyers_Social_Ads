# Social ads purchase prediction and classification optimization (Python)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit_Learn](https://img.shields.io/badge/-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-EB6512?style=for-the-badge&logo=xgboost&logoColor=white)

## Description of the project
This project focuses on data science and classification modeling applied to social media marketing campaigns. The main objective is to analyze user demographic indicators to predict purchasing behavior and heavily optimize targeting efficiency.

The analytical pipeline is specifically engineered to maximize the recall metric. In a targeted advertising context, failing to identify a potential buyer represents a direct loss of revenue, making the minimization of false negatives the primary operational priority.

**Dataset:** the analysis utilizes target campaign transaction records capturing user age, estimated annual salary levels, and final conversion outcomes.

## Technologies and libraries
* **Python**: core programming language for predictive pipeline development.
* **Scikit-learn**: machine learning framework used for preprocessing, scaling, baseline modeling, and validation.
* **XGBoost**: advanced gradient boosting library leveraged for high-performance classification.
* **Pandas / NumPy**: libraries utilized for structured array management and local statistical aggregations.
* **Matplotlib / Seaborn**: visualization engines employed for behavioral plotting and validation analysis.

## Key analysis phases

### 1. Exploratory data analysis and profiling
* **Demographic assessment**: analysis of data distributions across user age and estimated salary brackets to detect structural anomalies.
* **Behavioral correlation**: profiling conversion rates against user profiles to identify distinct purchasing thresholds and patterns.

### 2. Preprocessing and structural preparation
* **Feature scaling**: implementation of statistical standardization to adjust age and salary distributions for distance-based classification models.
* **Validation partitioning**: structuring data splits into independent training and testing subsets to protect final evaluations against data leakage.

### 3. Model training and hyperparameter optimization
* **Algorithm benchmarking**: initial training and evaluation of multiple classification paradigms including logistic regression, k-nearest neighbors, decision trees, and random forests.
* **Recall maximization**: execution of hyperparameter search routines specifically configured to penalize missing true buyers and isolate statistical outliers.

## How to run it?
1. Ensure you have **Python 3.x** installed along with a functional jupyter notebook environment.
2. Place the data file `Social_Ads_Recall.ipynb` in your working directory.
3. Install the required dependencies via your terminal:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn notebook
