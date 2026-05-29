# Social ads purchase prediction and classification optimization (Python)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit_Learn](https://img.shields.io/badge/-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-EB6512?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

## Description of the project
This project focuses on data science and classification modeling applied to social media marketing campaigns. The main objective is to analyze user demographic indicators to predict purchasing behavior and heavily optimize targeting efficiency.

The analytical pipeline is specifically engineered to maximize the recall metric. In a targeted advertising context, failing to identify a potential buyer represents a direct loss of revenue, making the minimization of false negatives the primary operational priority.

**Dataset:** the analysis utilizes target campaign transaction records capturing user age, estimated annual salary levels, and final conversion outcomes.

## Technologies and libraries
* **Python**: core programming language for predictive pipeline development.
* **Scikit-learn**: machine learning framework used for preprocessing, feature scaling, model training, and validation scoring.
* **XGBoost**: advanced gradient boosting library leveraged for high-performance classification.
* **Pandas / NumPy**: libraries utilized for structured array management, matrix operations, and local statistical aggregations.
* **Matplotlib / Seaborn**: visualization engines employed for behavioral plotting and validation analysis.

## Key analysis phases

### 1. Exploratory data analysis and profiling
* **Demographic assessment**: analysis of data distributions across user age and estimated salary brackets to detect structural anomalies.
* **Behavioral correlation**: profiling conversion rates against user profiles to identify distinct purchasing thresholds and patterns.

### 2. Preprocessing and structural preparation
* **Feature scaling**: implementation of standard scaling (`StandardScaler`) to adjust age and salary distributions, ensuring distance-based models are not statistically biased.
* **Validation partitioning**: structuring data splits using stratified methods to protect final evaluations against data leakage.

### 3. Model training and benchmark development
* **Algorithmic diversification**: training and evaluation of baseline performance metrics across distinct mathematical paradigms:
  * Linear models: logistic regression
  * Distance-based models: k-nearest neighbors (KNN) and support vector machines (SVM)
  * Tree-based models: decision trees and random forests
  * Boosting architectures: XGBoost

### 4. Hyperparameter optimization
* **Grid search execution**: implementation of exhaustive parameter grids (`GridSearchCV`) across target estimators.
* **Recall maximization**: cross-validation routines specifically configured to prioritize sensitivity scores, minimizing false negatives to retain target converting users.

## How to run it?
1. Ensure you have **Python 3.x** installed along with a functional jupyter notebook environment.
2. Place the data file `Social_Ads_Recall.ipynb` in your working directory.
3. Install the required dependencies via your terminal:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn notebook
4. Launch the jupyter interface and run the Social_Ads_Recall.ipynb cells sequentially to replicate the pipeline.

## Academic context
This project was developed as part of the Supervised Learning and Classification modules of the master's degree, showcasing proficiency in operational target optimization, algorithmic benchmarking, and performance metric alignment.

Developed by: Alicia Santamaría Román

Contact: [LinkedIn](https://linkedin.com/in/aliciasantamariaroman)
