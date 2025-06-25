#  Real-Time Credit Card Fraud Detection Using Machine Learning

##  Project Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions in real-time. By leveraging various supervised learning techniques, including ensemble models and anomaly detection methods, this system can effectively identify fraudulent behavior, which is critical for minimizing financial losses in the credit card industry.

---

##  Dataset

**Source:** [Kaggle - Credit Card Fraud Detection Dataset](http://kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)  
**Records:** 284,807 transactions  
**Features:** 30 anonymized features (V1, V2, ..., V28)

### Feature Details
- `Time`: Seconds since the first transaction
- `Amount`: Transaction amount (scaled)
- `Class`: Target variable (1 = fraud, 0 = legitimate)
- `V1–V28`: PCA-transformed, anonymized features

 The dataset is **highly imbalanced**, with fraud cases representing **less than 1%** of the total, making it ideal for testing techniques that handle class imbalance.

---

##  Technologies Used

- **Languages & Libraries**: Python, Pandas, NumPy, Scikit-Learn, XGBoost, imbalanced-learn (SMOTE), SHAP, Matplotlib, Seaborn  
- **Tools**: Jupyter Notebooks, Git, GitHub

---

##  Key Steps

- **Exploratory Data Analysis (EDA)**  
  Visualized fraud vs non-fraud distributions, analyzed transaction amounts and time patterns.

- **Data Preprocessing**  
  Handled missing values, scaled `Time` and `Amount`, removed outliers.

- **Class Imbalance Handling**  
  Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance classes.

- **Model Building**  
  Built and compared **Logistic Regression**, **Random Forest**, and **XGBoost**. Used **GridSearchCV** and **RandomizedSearchCV** for hyperparameter tuning.

- **Model Evaluation**  
  Evaluated models using **ROC-AUC**, **F1-Score**, **Recall**, **Precision**, and **Confusion Matrix**, focusing on high **recall** to minimize undetected fraud.

- **Model Explainability with SHAP**
  We used SHAP (SHapley Additive exPlanations) to interpret the XGBoost model and identify the features that most influence fraud predictions.

  The plot below summarizes the global importance and impact of each feature:

  ![image](https://github.com/user-attachments/assets/d66fe6b7-60f2-4a1c-b19c-933ead5d9a37)



  Each point represents a SHAP value for a transaction and a feature.

  Features are ranked by their mean absolute SHAP value (importance).

  Color shows the feature value (red = high, blue = low).
 
  X-axis shows how much each feature contributes to pushing the model’s output toward fraud or not fraud.

  Key Insights:

  V14, V4, V8, and V12 are the most influential in determining fraud.

  High values of V14 decrease the likelihood of fraud, while low values increase it.

  SHAP provides transparency, which is crucial for deploying models in sensitive financial domains.

---

**Model Performance Comparison**

We evaluated the models using the Precision-Recall Curve, which is particularly informative for highly imbalanced datasets like fraud detection, where the positive class (fraud) is rare.

![image](https://github.com/user-attachments/assets/ed02e6ba-45bc-4d31-879a-67b142cc280a)


  Key Insights:
  XGBoost consistently achieves the highest precision at most recall thresholds, indicating the most reliable fraud detection performance.

  Random Forest performs closely behind XGBoost, with a strong balance of recall and precision.

  Logistic Regression underperforms comparatively, showing lower precision across a range of recall values.

  Based on this analysis, XGBoost is selected as the optimal model due to its superior balance between catching fraudulent transactions (recall) and minimizing false positives (precision).

##  Results

- **XGBoost ROC-AUC Score**: **0.98** (excellent model discrimination)  
- **Recall**: **95%** (high sensitivity, critical for detecting fraud)  
- **Precision**: **98%** (high confidence in flagged fraud cases)

>  The model demonstrates strong performance in identifying fraud with minimal false negatives and is suitable for deployment in real-world financial systems.
